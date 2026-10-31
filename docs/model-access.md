# Model access and license inventory

Checked on 2026-10-30 on the reference Mac using unauthenticated Hugging Face
metadata requests. This is an access and license inventory, not a legal
interpretation or a guarantee of future access. Read the current terms before use
or redistribution.

| Component | Source | Published license | Access observed |
| --- | --- | --- | --- |
| Idol weights | [speridlabs/idol](https://huggingface.co/speridlabs/idol) | Apache-2.0, as reported by the model metadata; [upstream SDK license](https://github.com/speridlabs/idol-sdk/blob/main/LICENSE) | Public metadata requests succeeded for all five required files. |
| DINOv3 ViT-H+ | [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) | [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) as listed on the model card; review the current terms on the model page. | Gated, manual approval. An unauthenticated request for `model.safetensors` returned HTTP 401. |
| FLUX.2 dev VAE | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) | FLUX non-commercial license, as listed on the model card; consult the [repository license](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.txt) and access terms. | Gated, automatically approved after accepting the conditions. An unauthenticated request for `vae/diffusion_pytorch_model.safetensors` returned HTTP 401. |

The SDK uses the VAE from FLUX.2 dev. Do not assume that a different FLUX
model, its VAE, or a converted checkpoint has the same license or is
interchangeable. Review the model and component terms for the intended use and
distribution. The model card lists a non-commercial license for FLUX.2 dev;
verify commercial and production rights directly with the rights holder.

## Files checked

The original SDK requests these files from Idol: `config.json`,
`denoiser.safetensors`, `decoder.safetensors`, `decoder_gs.safetensors` and
`birefnet.safetensors`. Metadata for all five was accessible without a token.
This was a metadata check, not a full download or integrity verification.

No Hugging Face token was available to the checking environment. Therefore the
HTTP 401 responses establish unauthenticated access is blocked; they do not
establish whether any particular account has already been approved.

## Enable and verify access

1. Sign in to Hugging Face and review the conditions on the DINOv3 and FLUX.2
   model pages linked above. Request or accept access as appropriate; DINOv3
   may require approval.
2. Authenticate locally using `hf auth login`. Use a token with read access
   authorized for gated repositories. Do not place tokens in issues, logs,
   commits, or documentation.
3. Check the active account with `hf auth whoami`. Repeat the access check
   from the reference environment after access has been granted. A network
   error or rate limit does not prove a license or access denial.
4. Keep model weights in a local cache and respect each model's terms. This
   repository does not bundle them.

Full checkpoint downloads, checksum validation and inference are not covered by
this inventory. Issue #6 can be resolved by identifying missing access; obtaining
approved access is a prerequisite for later full-model validation.

Spanish: [es/model-access.md](../es/model-access.md).
