import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_asr.model as module_model
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.metric.utils import calc_wer, calc_cer

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"We are running on {device}")

    # text_encoder
    text_encoder = config.get_text_encoder()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    results = []
    cer_argmax = []
    cer_bs = []
    cer_bs_lm = []

    wer_argmax = []
    wer_bs = []
    wer_bs_lm = []

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)
            output = model(**batch)
            if type(output) is dict:
                batch.update(output)
            else:
                batch["logits"] = output
            batch["log_probs"] = torch.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["probs"] = batch["log_probs"].exp().cpu()
            probs = batch["probs"]
            batch["argmax"] = batch["probs"].argmax(-1)

            for i in range(len(batch["text"])):
                argmax = batch["argmax"][i]
                argmax = argmax[: int(batch["log_probs_length"][i])]
                target_text = batch["text"][i]
                argmax_pred = text_encoder.ctc_decode(argmax.cpu().numpy())
                
                bs_preds = text_encoder.ctc_beam_search(
                            probs[i], batch["log_probs_length"][i], beam_size=10
                        )[:10]
                bs_lm_preds = text_encoder.ctc_beam_search_lm(
                            probs[i], batch["log_probs_length"][i], beam_size=512
                        )[0]

                cer_argmax.append(calc_cer(target_text, argmax_pred))
                wer_argmax.append(calc_wer(target_text, argmax_pred))

                cer_bs.append(calc_cer(target_text, bs_preds[0].text))
                wer_bs.append(calc_wer(target_text, bs_preds[0].text))

                cer_bs_lm.append(calc_cer(target_text, bs_lm_preds.text))
                wer_bs_lm.append(calc_wer(target_text, bs_lm_preds.text))

                results.append(
                    {
                        "ground_truth": target_text,
                        "pred_text_argmax": argmax_pred,
                        "pred_text_beam_search": [hypo.text for hypo in bs_preds],
                        "pred_text_beam_search_LM": bs_lm_preds.text,
                    }
                )
    print("Final_metrics")
    print("  Argmax:")
    print(f"    CER: {sum(cer_argmax) / len(cer_argmax)}")
    print(f"    WER: {sum(wer_argmax) / len(wer_argmax)}")
    print("  Custom BeamSearch:")
    print(f"    CER: {sum(cer_bs) / len(cer_bs)}")
    print(f"    WER: {sum(wer_bs) / len(wer_bs)}")
    print("  BeamSearch+LM:")
    print(f"    CER: {sum(cer_bs_lm) / len(cer_bs_lm)}")
    print(f"    WER: {sum(wer_bs_lm) / len(wer_bs_lm)}")
    with Path(out_file).open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder / "audio"),
                            "transcription_dir": str(
                                test_data_folder / "transcriptions"
                            ),
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
