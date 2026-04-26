from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT_CANDIDATES = [
    PROJECT_ROOT / "models" / "trainModelsAI" / "train_vit5_summarize.py",
    PROJECT_ROOT / "scripts" / "train_vit5_summarize.py",
]


def resolve_train_script() -> Path:
    for path in TRAIN_SCRIPT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Khong tim thay script train. Da tim tai:\n"
        + "\n".join(f"- {candidate}" for candidate in TRAIN_SCRIPT_CANDIDATES)
    )


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    default_label = "Y/n" if default else "y/N"
    raw = input(f"{prompt} ({default_label}): ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1"}


def ask_int(prompt: str, default: int | None = None) -> int | None:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print("Gia tri khong hop le, bo qua.")
        return default


def ask_float(prompt: str, default: float | None = None) -> float | None:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("Gia tri khong hop le, bo qua.")
        return default


def build_command(
    train_script: Path,
    lang: str,
    resume: bool = False,
    cpu: bool = False,
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    cmd: list[str] = [sys.executable, str(train_script), "--lang", lang]
    if resume:
        cmd.append("--resume")
    if cpu:
        cmd.append("--cpu")

    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
                continue
            cmd.extend([f"--{key}", str(value)])
    return cmd


def run_train_job(
    train_script: Path,
    lang: str,
    resume: bool = False,
    cpu: bool = False,
    overrides: dict[str, Any] | None = None,
) -> None:
    cmd = build_command(
        train_script=train_script,
        lang=lang,
        resume=resume,
        cpu=cpu,
        overrides=overrides,
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    print("\nDang chay lenh:")
    print(" ".join(cmd))
    print("-" * 72)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)
    print("-" * 72)
    print(f"Hoan tat train lang={lang}, resume={resume}.")


def run_pair(
    train_script: Path,
    resume: bool = False,
    cpu: bool = False,
    overrides: dict[str, Any] | None = None,
) -> None:
    run_train_job(train_script, "vi", resume=resume, cpu=cpu, overrides=overrides)
    run_train_job(train_script, "en", resume=resume, cpu=cpu, overrides=overrides)


def prompt_overrides_for_4gb() -> dict[str, Any]:
    print("\nNhap tuy chinh train (Enter de dung mac dinh script):")
    print("- Goi y GPU 4GB: batch_size=1, max_input=512, max_target=128, epochs=2-4")
    overrides: dict[str, Any] = {}
    overrides["batch_size"] = ask_int("batch_size")
    overrides["epochs"] = ask_float("epochs")
    overrides["lr"] = ask_float("learning_rate")
    overrides["max_input"] = ask_int("max_input")
    overrides["max_target"] = ask_int("max_target")
    return overrides


def choose_data_profile() -> dict[str, Any]:
    openrouter = PROJECT_ROOT / "data" / "summary_data_openrouter.jsonl"
    train_processed = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
    val_processed = PROJECT_ROOT / "data" / "processed" / "val.jsonl"

    print("\nChon nguon du lieu:")
    print("1) Processed train/val (mac dinh)")
    print("2) OpenRouter train + val processed (khuyen nghi de fine-tune)")
    print("3) OpenRouter only, KHONG val (train nhanh)")
    choice = input("Lua chon [1]: ").strip() or "1"

    if choice == "2":
        return {
            "train_path": str(openrouter),
            "val_path": str(val_processed),
            "disable_eval": False,
        }
    if choice == "3":
        return {
            "train_path": str(openrouter),
            "disable_eval": True,
        }
    return {
        "train_path": str(train_processed),
        "val_path": str(val_processed),
        "disable_eval": False,
    }


def choose_text_clean_mode() -> str:
    print("\nLam sach text train:")
    print("1) keep (giu nguyen)")
    print("2) strip_headers (bo dong Tu khoa/Tieu de/Noi dung)")
    print("3) content_only (chi lay phan sau 'Noi dung:') [Khuyen nghi cho data ban]")
    choice = input("Lua chon [3]: ").strip() or "3"
    if choice == "1":
        return "keep"
    if choice == "2":
        return "strip_headers"
    return "content_only"


def print_overrides(overrides: dict[str, Any]) -> None:
    print("\nCau hinh se chay:")
    for key in sorted(overrides.keys()):
        print(f"- {key}: {overrides[key]}")


def choose_recommended_profile() -> dict[str, Any] | None:
    openrouter = PROJECT_ROOT / "data" / "summary_data_openrouter.jsonl"
    train_processed = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
    val_processed = PROJECT_ROOT / "data" / "processed" / "val.jsonl"

    print("\nChon preset toi uu:")
    print("1) Fine-tune OpenRouter + val (Khuyen nghi cho ban)")
    print("2) Base train processed + val")
    print("3) OpenRouter nhanh (khong val)")
    print("0) Khong dung preset")
    choice = input("Lua chon [1]: ").strip() or "1"

    if choice == "0":
        return None
    if choice == "2":
        return {
            "batch_size": 1,
            "epochs": 3,
            "lr": 1e-5,
            "max_input": 512,
            "max_target": 128,
            "train_path": str(train_processed),
            "val_path": str(val_processed),
            "disable_eval": False,
            "text_clean_mode": "content_only",
            "resume_mode": "weights_only",
            "resume_additional_epochs": 1.0,
        }
    if choice == "3":
        return {
            "batch_size": 1,
            "epochs": 1.0,
            "lr": 1e-5,
            "max_input": 512,
            "max_target": 128,
            "train_path": str(openrouter),
            "disable_eval": True,
            "text_clean_mode": "content_only",
            "resume_mode": "weights_only",
            "resume_additional_epochs": 1.0,
        }
    return {
        "batch_size": 1,
        "epochs": 3,
        "lr": 1e-5,
        "max_input": 512,
        "max_target": 128,
        "train_path": str(openrouter),
        "val_path": str(val_processed),
        "disable_eval": False,
        "text_clean_mode": "content_only",
        "resume_mode": "weights_only",
        "resume_additional_epochs": 1.0,
    }


def print_menu() -> None:
    print("\n================ TRAIN MENU ================")
    print("1) Train tieng Viet (moi)")
    print("2) Train tieng Anh (moi)")
    print("3) Train ca 2 (moi)")
    print("4) Train tiep tuc tieng Viet (resume)")
    print("5) Train tiep tuc tieng Anh (resume)")
    print("6) Train tiep tuc ca 2 (resume)")
    print("7) Custom (chon nguon du lieu + clean mode + resume + tham so)")
    print("0) Thoat")
    print("===========================================")


def handle_custom(train_script: Path) -> None:
    lang_raw = input("Chon ngon ngu (vi/en/both) [vi]: ").strip().lower() or "vi"
    if lang_raw not in {"vi", "en", "both"}:
        print("Lua chon khong hop le.")
        return

    resume = ask_yes_no("Resume tu checkpoint gan nhat?", default=False)
    cpu = ask_yes_no("Force train bang CPU?", default=False)
    use_recommended = ask_yes_no("Dung preset toi uu de do hoi nhieu?", default=True)
    overrides: dict[str, Any]
    if use_recommended:
        picked = choose_recommended_profile()
        if picked is None:
            overrides = prompt_overrides_for_4gb()
            overrides.update(choose_data_profile())
            overrides["text_clean_mode"] = choose_text_clean_mode()
        else:
            overrides = picked
    else:
        overrides = prompt_overrides_for_4gb()
        overrides.update(choose_data_profile())
        overrides["text_clean_mode"] = choose_text_clean_mode()
        if not overrides.get("disable_eval", False):
            disable_eval_quick = ask_yes_no(
                "Bo qua evaluate/val de train nhanh hon?",
                default=False,
            )
            if disable_eval_quick:
                overrides["disable_eval"] = True
                overrides.pop("val_path", None)

    if resume:
        if "resume_mode" not in overrides:
            mode_raw = input("resume_mode (weights_only/stateful) [weights_only]: ").strip().lower() or "weights_only"
            if mode_raw not in {"weights_only", "stateful"}:
                mode_raw = "weights_only"
            overrides["resume_mode"] = mode_raw
        if "resume_additional_epochs" not in overrides:
            overrides["resume_additional_epochs"] = ask_float(
                "resume_additional_epochs",
                default=1.0,
            )
    else:
        overrides.pop("resume_mode", None)
        overrides.pop("resume_additional_epochs", None)

    print_overrides(overrides)
    if not ask_yes_no("Bat dau train voi cau hinh tren?", default=True):
        print("Huy theo yeu cau.")
        return

    if lang_raw == "both":
        run_pair(train_script, resume=resume, cpu=cpu, overrides=overrides)
    else:
        run_train_job(train_script, lang_raw, resume=resume, cpu=cpu, overrides=overrides)


def main() -> None:
    try:
        train_script = resolve_train_script()
    except FileNotFoundError as exc:
        print(str(exc))
        raise SystemExit(1)

    print("Su dung script train:", train_script)
    while True:
        print_menu()
        choice = input("Nhap lua chon: ").strip()
        try:
            if choice == "1":
                run_train_job(train_script, "vi", resume=False, cpu=False)
            elif choice == "2":
                run_train_job(train_script, "en", resume=False, cpu=False)
            elif choice == "3":
                run_pair(train_script, resume=False, cpu=False)
            elif choice == "4":
                run_train_job(train_script, "vi", resume=True, cpu=False)
            elif choice == "5":
                run_train_job(train_script, "en", resume=True, cpu=False)
            elif choice == "6":
                run_pair(train_script, resume=True, cpu=False)
            elif choice == "7":
                handle_custom(train_script)
            elif choice == "0":
                print("Da thoat train menu.")
                return
            else:
                print("Lua chon khong hop le, thu lai.")
        except subprocess.CalledProcessError as exc:
            print(f"\nTrain bi loi (exit code {exc.returncode}).")
        except KeyboardInterrupt:
            print("\nDa dung theo yeu cau nguoi dung.")
            return


if __name__ == "__main__":
    main()
