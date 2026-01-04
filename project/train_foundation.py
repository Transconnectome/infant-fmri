import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from module.models.titans_neuro import TitansNeuro
from module.utils.data_module import fMRIDataModule
try:
    import neptune.new as neptune
except ImportError:
    neptune = None

class LitTitans(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = TitansNeuro(
            img_size=self.hparams.img_size,
            in_chans=self.hparams.in_chans,
            embed_dim=self.hparams.embed_dim,
            window_size=self.hparams.window_size,
            first_window_size=self.hparams.first_window_size,
            patch_size=self.hparams.patch_size,
            depths=self.hparams.depths,
            num_heads=self.hparams.num_heads,
            c_multiplier=self.hparams.c_multiplier,
            last_layer_full_MSA=self.hparams.last_layer_full_MSA,
            drop_rate=self.hparams.drop_rate,
            attn_drop_rate=self.hparams.attn_drop_rate,
            drop_path_rate=self.hparams.drop_path_rate,
            num_subjects=self.hparams.num_subjects,
            use_memory=self.hparams.use_memory
        )
        
    def forward(self, x, subject_ids=None):
        return self.model(x, subject_ids)
    
    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x = batch['fmri_sequence']
        else:
             x = batch[0]
        
        # x shape: (B, C, D, H, W, T)
        # 1. Compute Target Features (Unmasked)
        with torch.no_grad():
            # Ideally use a separate target encoder (EMA), but for MVP we use the same model in eval mode or just without mask?
            # SwiFT/Titans encoder is deterministic typically (unless dropout).
            # We want to reconstruct the *features* of the unmasked input.
            # But TitansNeuro.forward does pooling.
            y_target = self.model(x).detach()

        # 2. Apply Masking to Input
        B, C, D, H, W, T = x.shape
        # Create mask: (B, T)
        mask_ratio = 0.5
        noise = torch.rand(B, T, device=x.device)
        # Mask 50% of frames
        mask = noise < mask_ratio # (B, T) boolean
        
        # Apply mask to x
        # x: (B, C, D, H, W, T)
        # mask needs to be broadcastable
        mask_bc = mask.view(B, 1, 1, 1, 1, T)
        x_masked = x * (~mask_bc) # Zero out masked frames
        
        # 3. Forward Masked Input
        y_hat = self.model(x_masked)
        
        # 4. Compute Loss
        # y_hat, y_target are (B, T, C_out)
        # distinct from 0? masked frames should be predicted.
        # Loss only on masked patches or all? BERT/MAE usually on masked only.
        
        loss = F.mse_loss(y_hat[mask], y_target[mask])
        
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("TitansNeuro")
        # group.add_argument("--img_size", nargs="+", default=[96, 96, 96, 20], type=int) # Defined in DataModule
        group.add_argument("--in_chans", type=int, default=1)
        group.add_argument("--embed_dim", type=int, default=24)
        group.add_argument("--window_size", nargs="+", default=[4, 4, 4, 4], type=int)
        group.add_argument("--first_window_size", nargs="+", default=[2, 2, 2, 2], type=int)
        group.add_argument("--patch_size", nargs="+", default=[6, 6, 6, 1], type=int)
        group.add_argument("--depths", nargs="+", default=[2, 2, 6, 2], type=int)
        group.add_argument("--num_heads", nargs="+", default=[3, 6, 12, 24], type=int)
        group.add_argument("--c_multiplier", type=int, default=2)
        group.add_argument("--last_layer_full_MSA", action='store_true')
        group.add_argument("--drop_rate", type=float, default=0.0)
        group.add_argument("--attn_drop_rate", type=float, default=0.0)
        group.add_argument("--drop_path_rate", type=float, default=0.1)
        group.add_argument("--num_subjects", type=int, default=1000)
        group.add_argument("--use_memory", action='store_true')
        group.add_argument("--learning_rate", type=float, default=1e-4) # Added LR
        return parser

def main():
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    
    # Trainer args
    # parser = pl.Trainer.add_argparse_args(parser) # Deprecated in PL 2.0
    group = parser.add_argument_group("Trainer")
    group.add_argument("--accelerator", type=str, default="auto")
    group.add_argument("--devices", type=int, default=1)
    group.add_argument("--max_epochs", type=int, default=10)
    group.add_argument("--limit_train_batches", type=float, default=1.0)
    group.add_argument("--limit_val_batches", type=float, default=1.0)
    group.add_argument("--limit_test_batches", type=float, default=1.0)
    group.add_argument("--check_val_every_n_epoch", type=int, default=1)
    
    # Model args
    parser = LitTitans.add_model_specific_args(parser)
    
    # Data args
    parser = fMRIDataModule.add_data_specific_args(parser)
    
    # General args
    parser.add_argument("--project_name", type=str, default="titans-neuro")
    parser.add_argument("--experiment_name", type=str, default="pretrain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loggername", type=str, default="tensorboard", choices=["neptune", "tensorboard"])
    parser.add_argument("--pretraining", action='store_true', default=True) # Force pretraining mode
    parser.add_argument("--dataset_name", type=str, default="Dummy", choices=["S1200", "ABCD", "UKB", "dHCP", "Dummy", "Music", "Narratives"])
    
    # Missing args required by fMRIDataModule (originally in LitClassifier)
    parser.add_argument("--use_contrastive", action='store_true')
    parser.add_argument("--contrastive_type", type=int, default=0)
    parser.add_argument("--downstream_task", type=str, default="None")

    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    
    # DataModule
    dm = fMRIDataModule(**vars(args))
    
    # Model
    model = LitTitans(**vars(args))
    
    # Logger
    if args.loggername == "neptune":
        api_key = os.environ.get("NEPTUNE_API_TOKEN")
        logger = NeptuneLogger(api_key=api_key, project=args.project_name, name=args.experiment_name)
    else:
        logger = TensorBoardLogger("output", name=args.experiment_name)
        
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        mode='min',
        save_last=True,
        filename='titans-{epoch:02d}-{train_loss:.2f}'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
    )
    
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()
