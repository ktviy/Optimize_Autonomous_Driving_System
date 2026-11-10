from pathlib import Path
import torch


ROOT = Path(__file__).resolve().parent.parent
SECTIONS = [
    ("SNN_T", ROOT / "SNN_T" / "weights"),
    ("Transformer", ROOT / "Transformer" / "weights"),
]


def format_metric(value):
    if value is None:
        return "not found"
    if isinstance(value, float):
        return f"{value:.10f}".rstrip('0').rstrip('.') if '.' in f"{value:.10f}" else str(value)
    return str(value)


lines = []
for name, folder in SECTIONS:
    lines.append("=" * 60)
    lines.append(f"{name} WEIGHTS")
    lines.append("=" * 60)

    if folder.exists():
        pth_files = sorted([p for p in folder.glob("*.pth") if p.is_file()])
        if pth_files:
            lines.append(f"Total .pth files: {len(pth_files)}")
            for path in pth_files:
                try:
                    try:
                        data = torch.load(path, map_location="cpu", weights_only=False)
                    except TypeError:
                        data = torch.load(path, map_location="cpu")
                except Exception as exc:
                    lines.append(f"  Could not load checkpoint: {exc}")
                    continue

                epoch = data.get('epoch')
                train_loss = data.get('train_loss')
                val_acc = data.get('val_acc')

                if torch.is_tensor(epoch):
                    epoch = epoch.item() if epoch.numel() == 1 else None
                if torch.is_tensor(train_loss):
                    train_loss = train_loss.item() if train_loss.numel() == 1 else None
                if torch.is_tensor(val_acc):
                    val_acc = val_acc.item() if val_acc.numel() == 1 else None

                if epoch is not None and train_loss is not None and val_acc is not None:
                    lines.append(f"  Epoch {epoch} | Loss: {train_loss:.4f} | Val Accuracy: {val_acc * 100:.2f}%")
                else:
                    lines.append(f"  Epoch {epoch if epoch is not None else 'not found'} | Loss: {format_metric(train_loss)} | Val Accuracy: {format_metric(val_acc)}")
        else:
            lines.append("(No .pth files found)")
    else:
        lines.append(f"(Folder not found: {folder})")

    lines.append("")

content = "\n".join(lines).rstrip() + "\n"
print(content, end="")

log_path = ROOT / "logs" / "train_data_log.txt"
log_path.write_text(content, encoding="utf-8")
