from models.sds import StableDiffusion
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def fix_randomness(seed=42):
    import os
    import numpy as np
    # https: // www.zhihu.com/question/542479848/answer/2567626957
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def zero_shot_gen_sds():
    # optimize 2d tensor as image representation.
    import PIL.Image as Image
    device = torch.device('cuda')
    # image = nn.Parameter(torch.empty(1, 3, 512, 512, device=device))
    # #init with the image to be edited
    # with open("panda_snow.png", 'rb') as f:
    #     image_ = Image.open(f)
    #     image_ = torchvision.transforms.functional.resize(image_, (512, 512))
    #     image_ = torchvision.transforms.functional.to_tensor(image_)[:3,...].unsqueeze(0)
    #     image.data = image_.to(device)
    
    latent = nn.Parameter(torch.empty(1, 4, 64, 64, device=device))
    latent.data = torch.randn_like(latent.data)
    optimizer = torch.optim.SGD([latent], lr=1.0)
    decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
    n_iter = 200
    sd = StableDiffusion(device, version="2.0")
    image = sd.decode_latents(latent)
    prompt = "a Squirrel is snowboarding"
    image_steps = []
    for i in range(n_iter):
        optimizer.zero_grad()
        sd.manual_backward(sd.get_text_embeds(prompt), image, guidance_scale=20, latent=latent)
        optimizer.step()
        if i % 20 == 0:
            decay.step()
            print(f'[INFO] iter {i}, loss {torch.norm(latent.grad)}')
            image = sd.decode_latents(latent)
            image_steps.append(image.detach().clone())
    
    # visualize as grid
    image_steps = torch.cat(image_steps, dim=0)
    from torchvision.utils import make_grid
    grid = make_grid(image_steps, nrow=5, padding=10)
    # save
    from torchvision.utils import save_image
    save_image(grid, 'image_steps.png')

def edit_sds():
    # optimize 2d tensor as image representation.
    import PIL.Image as Image
    device = torch.device('cuda')


    sd = StableDiffusion(device, version="2.0")
    latent = nn.Parameter(torch.empty(1, 4, 64, 64, device=device))
    #init with the image to be edited
    with open("/root/autodl-tmp/gaussian-splatting/data/flower/images/image000.png", 'rb') as f:
        image_ = Image.open(f)
        image_ = torchvision.transforms.functional.resize(image_, (512, 512))
        image_ = torchvision.transforms.functional.to_tensor(image_)[:3,...].unsqueeze(0)
        image_latent = sd.encode_imgs(image_.to(device))
        latent.data = image_latent.data
    optimizer = torch.optim.SGD([latent], lr=0.2)
    decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
    n_iter = 50

    image = sd.decode_latents(latent)
    prompt = "a yellow flower"
    image_steps = []
    for i in range(n_iter):
        optimizer.zero_grad()
        sd.manual_backward(sd.get_text_embeds(prompt), image, guidance_scale=20, latent=latent)
        optimizer.step()
        if i % 10 == 0:
            decay.step()
            print(f'[INFO] iter {i}, loss {torch.norm(latent.grad)}')
            image = sd.decode_latents(latent)
            image_steps.append(image.detach().clone())
    
    # visualize as grid
    image_steps = torch.cat(image_steps, dim=0)
    from torchvision.utils import make_grid
    grid = make_grid(image_steps, nrow=5, padding=10)
    # save
    from torchvision.utils import save_image
    save_image(grid, 'image_steps.png')

def edit_dds():
    from stylize import VGGPerceptualLoss
    # optimize 2d tensor as image representation.
    import PIL.Image as Image
    device = torch.device('cuda')


    sd = StableDiffusion(device, version="2.0")
    latent = nn.Parameter(torch.empty(1, 4, 64, 64, device=device))
    #init with the image to be edited
    with open("/root/autodl-tmp/gaussian-splatting/data/trex/images/DJI_20200223_163548_810.png", 'rb') as f:
        image_ = Image.open(f)
        image_ = torchvision.transforms.functional.resize(image_, (512, 512))
        image_ = torchvision.transforms.functional.to_tensor(image_)[:3,...].unsqueeze(0)
        image_latent = sd.encode_imgs(image_.to(device))
        latent.data = image_latent.data
        latent_orig = latent.data.clone().requires_grad_(False)
    optimizer = torch.optim.SGD([latent], lr=1.0)
    decay = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.9)
    n_iter = 100

    image = sd.decode_latents(latent)
    prompt_a = "a DSLR photo"
    prompt_b = "a pencil sketch"
    image_steps = []
    for i in range(n_iter):
        optimizer.zero_grad()
        # grad_a = sd.calc_grad(sd.get_text_embeds(prompt_a), image, guidance_scale=7.5, latent=latent_orig)
        # grad_b = sd.calc_grad(sd.get_text_embeds(prompt_b), image, guidance_scale=7.5, latent=latent)
        # grad_apply = grad_b - grad_a
        # latent.backward(gradient=grad_apply, retain_graph=True)
        sd.manual_backward_dds(sd.get_text_embeds(prompt_a), image, sd.get_text_embeds(prompt_b), image, guidance_scale=7.5, src_latent=latent_orig, tgt_latent=latent)
        optimizer.step()
        decay.step()
        if i % 20 == 0:
            print(f'[INFO] iter {i}, loss {torch.norm(latent.grad)}')
            image = sd.decode_latents(latent)
            image_steps.append(image.detach().clone())
    
    # visualize as grid
    image_steps = torch.cat(image_steps, dim=0)
    from torchvision.utils import make_grid
    grid = make_grid(image_steps, nrow=5, padding=10)
    # save
    from torchvision.utils import save_image
    save_image(grid, 'image_steps.png')

fix_randomness()
# edit_sds()
edit_dds()
