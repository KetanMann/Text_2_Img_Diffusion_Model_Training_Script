from diffusers import DiffusionPipeline
import torch

class TextToImagePipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler, text_encoder, tokenizer):
        super().__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer
        )

    def __call__(self, prompt, num_inference_steps=50, generator=None, output_type="pil"):
        # Encode text
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Create random noise
        latents = torch.randn(
            (1, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
            device=self.device
        )

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Denoising loop
        for t in self.progress_bar(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = latents.clamp(-1, 1)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image}