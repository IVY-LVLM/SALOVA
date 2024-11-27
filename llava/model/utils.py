from transformers import AutoConfig
import torch
import random

def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if "llava" in config and "llava" not in cfg.model_type:
        assert cfg.model_type == "llama"
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = "LlavaLlamaForCausalLM"
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)

def get_loss_fn(resampler_cls, sim_score, loss_fn, weight_ce, beta=0.2, train_step='lv'):
    sim_loss = 0.0
    total_correct = 0
    total_samples = 0
    correctv2 = 0

    batch_size = len(resampler_cls)
   
    for cls_logits, sim in zip(resampler_cls, sim_score):
        clip_num = cls_logits.size(0)
        bce_loss = loss_fn(cls_logits, sim)  # Scalar
        if train_step == 'lv' or train_step == 'sft':
            # Get indices of positive and negative samples
            pos_indices = torch.nonzero(sim.view(-1).float() == 1).view(-1)
            neg_indices = torch.nonzero(sim.view(-1).float() == 0).view(-1)
            
            # Determine the number of samples to take
            num_pos = len(pos_indices)
            num_neg = len(neg_indices)
            sample_num = min(num_pos, num_neg)
            
            if sample_num > 0:
                rand_num = random.randint(1, sample_num)
                # Randomly sample indices
                sampled_pos_indices = pos_indices[torch.randperm(num_pos)[:rand_num]]
                sampled_neg_indices = neg_indices[torch.randperm(num_neg)[:rand_num]]
                
                # Subset logits and labels
                pos_cls_logits = cls_logits.view(-1)[sampled_pos_indices]
                neg_cls_logits = cls_logits.view(-1)[sampled_neg_indices]

                margin_loss = torch.clamp(beta + neg_cls_logits - pos_cls_logits, min=0).sum() / rand_num
                sim_loss += margin_loss

        sim_loss += bce_loss
        # # Compute predictions and accuracy
        # preds = (torch.sigmoid(cls_logits) > 0.5).float()
        # total_correct += (preds == sim).sum().item()
        # total_samples += clip_num

        # acc v2
        preds = torch.argmax(cls_logits, dim=1)
        correct_predictions = sim[torch.arange(sim.size(0)), preds]

        correctv2 += correct_predictions.sum()

    current_acc = correctv2 / clip_num
    if train_step == 'sft':
        current_acc = correctv2 / batch_size
    
    # Average loss and accuracy over the batch
    sim_loss = weight_ce * (sim_loss / batch_size)
    # total_acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    return sim_loss, current_acc, correctv2

def get_contrastive_loss(resampler_cls, sim_score, loss_fn, weight_ce, beta=0.2):
    sim_loss = 0.0
    total_correct = 0
    total_samples = 0
    correctv2 = 0

    batch_size = len(resampler_cls)
                
    for cls_logits, sim in zip(resampler_cls, sim_score):
        clip_num = cls_logits.size(0)

        loss_text = loss_fn(cls_logits, sim)  # Scalar
        loss_image = loss_fn(cls_logits.t(), sim)  # Scalar

        sim_loss = (loss_text + loss_image) / 2

        # Compute predictions and accuracy
        total_correct += (preds == sim).sum().item()
        total_samples += clip_num
        # acc v2
        preds = torch.argmax(cls_logits, dim=1)
        gts = torch.argmax(sim, dim=1)

        correctv2 += (preds == gts).sum().float()
        current_acc = (preds == gts).sum().float() / clip_num

    # Average loss and accuracy over the batch
    sim_loss = weight_ce * (sim_loss / batch_size)
    # total_acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    # return sim_loss, total_acc, correctv2
    return sim_loss, current_acc, correctv2