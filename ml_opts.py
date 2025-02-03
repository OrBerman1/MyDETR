import logging
from coco_eval import CocoEvaluator
import torch
import torch.nn.functional as F
from metrics import calculate_ap_and_map
from checkpoint import save_checkpoint
from tqdm import tqdm as tqdm
from metrics import prepare_for_coco_detection, calculate_ap


def train_one_epoch(model, data_loader, optimizer, scaler, device, enable_amp):
    model.train()
    total_loss = 0
    for pixel_values, targets in tqdm(data_loader, total=len(data_loader)):
        # images = list(image.to(device) for image in images)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        padded = data_loader.dataset.processor.pad(pixel_values, return_tensors="pt")
        pixel_values = padded['pixel_values']
        masks = padded['pixel_mask']
        labels = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.autocast(device_type=str(device), dtype=torch.float16, enabled=enable_amp):
            outputs = model(pixel_values=pixel_values.to(device), pixel_mask=masks.to(device), labels=labels)
            loss = outputs.loss
        total_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    return total_loss / len(data_loader)


# Testing Loop
def evaluate(model, data_loader, device):
    model.eval()
    pred_bboxes = []
    pred_probs = []
    pred_labels = []
    all_targets = []
    with torch.no_grad():
        for pixel_values, targets in tqdm(data_loader, total=len(data_loader)):
            processed_targets = [{"boxes": t["boxes"].to(device), "labels": t["class_labels"].to(device)} for t in targets]
            all_targets += processed_targets
            padded = data_loader.dataset.processor.pad(pixel_values, return_tensors="pt")
            pixel_values = padded['pixel_values']
            masks = padded['pixel_mask']
            labels = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(pixel_values=pixel_values.to(device), pixel_mask=masks.to(device), labels=labels)
            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
            outputs = data_loader.dataset.processor.post_process_object_detection(outputs,
                                                                                  target_sizes=orig_target_sizes,
                                                                                  threshold=0)
            boxes = [d["boxes"] for d in outputs]
            labels = [d["labels"] for d in outputs]
            probs = [d["scores"] for d in outputs]
            # Get the predicted labels by taking the argmax over the class probabilities
            pred_bboxes += boxes
            pred_probs += probs
            pred_labels += labels
    return calculate_ap(pred_bboxes, pred_probs, pred_labels, all_targets)


def train(model, train_loader, test_loader, optimizer, scaler, start_epoch, args):
    best_map = 0
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, args.device, args.enable_amp)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {train_loss:.4f}")
        MAP, _ = evaluate(model, test_loader, args.device)
        # MAP = test_results["map"]

        # Save checkpoint every 'save_every' epochs
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(epoch, model, optimizer, scaler,
                            f"{args.experiments_path}/{args.experiment_name}/last.ckpt")

        if MAP >= best_map:
            save_checkpoint(epoch, model, optimizer, scaler,
                            f"{args.experiments_path}/{args.experiment_name}/best.ckpt")
            best_map = MAP

    # Final save
    save_checkpoint(args.epochs, model, optimizer, scaler, f"{args.experiments_path}/{args.experiment_name}/last.ckpt")

    # Evaluate on Test Set
    print("Testing complete!")
