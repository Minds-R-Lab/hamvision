#!/usr/bin/env python3
"""Diagnose HamSeg intermediate signals - print statistics to understand what's happening."""
import os, sys, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
warnings.filterwarnings('ignore')

def load_model(model_path, device, embed_dim=48):
    # Import hamseg from same directory as THIS script (not model_path)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if 'hamseg' in sys.modules:
        del sys.modules['hamseg']  # force reimport
    from hamseg import HamSeg
    class A: pass
    a = A(); a.embed_dim=embed_dim; a.depths=[2,2,2,2]; a.damping_clamp=5.0; a.num_classes=1; a.img_size=224; a.drop_rate=0.1
    model = HamSeg(a).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def load_one(data_root, img_size=224):
    root = Path(data_root)
    for split in ['test','val','train']:
        id_, md_ = root/split/'images', root/split/'masks'
        if id_.exists() and md_.exists():
            exts = {'.png','.jpg','.jpeg'}
            sfx = ['_segmentation','_Segmentation','_mask','_seg']
            ml = {}
            for p in md_.iterdir():
                if p.suffix.lower() in exts:
                    ml[p.stem] = p
                    for s in sfx:
                        if p.stem.endswith(s): ml[p.stem[:-len(s)]] = p
            for p in sorted(id_.iterdir()):
                if p.suffix.lower() not in exts: continue
                m = ml.get(p.stem)
                if not m:
                    for s in sfx:
                        m = ml.get(p.stem+s)
                        if m: break
                if m:
                    img = Image.open(str(p)).convert('RGB')
                    msk = Image.open(str(m)).convert('L')
                    t = TF.to_tensor(TF.resize(img, [img_size, img_size]))
                    t = TF.normalize(t, [.485,.456,.406], [.229,.224,.225])
                    mt = (torch.from_numpy(np.array(TF.resize(msk,[img_size,img_size]))).float() > 128).float()
                    return t, mt
    raise RuntimeError("No data found")

def stat(name, t):
    """Print statistics of a tensor."""
    a = t.float()
    print(f"  {name:30s} shape={str(list(a.shape)):15s} "
          f"min={a.min().item():10.4f}  max={a.max().item():10.4f}  "
          f"mean={a.mean().item():10.4f}  std={a.std().item():10.4f}  "
          f"zeros%={100*(a.abs()<1e-6).float().mean().item():.1f}")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='./outputs_hamseg/best_model.pth')
    p.add_argument('--data_root', required=True)
    p.add_argument('--embed_dim', type=int, default=48)
    a = p.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(a.model_path, device, a.embed_dim)
    img, msk = load_one(a.data_root)
    
    print("="*80)
    print("HamSeg Signal Diagnostics")
    print("="*80)
    
    with torch.no_grad():
        x = img.unsqueeze(0).to(device)
        
        # Encoder
        x_stem = model.stem(x)
        e1 = model.enc1(x_stem)
        e2 = model.enc2(model.down1(e1))
        e3 = model.enc3(model.down2(e2))
        e4 = model.down3(e3)
        
        print("\n--- Encoder outputs ---")
        stat("e1", e1)
        stat("e2", e2)
        stat("e3", e3)
        stat("e4 (into bottleneck)", e4)
        
        # Bottleneck - step through each sub-block
        print("\n--- Bottleneck internals ---")
        for bi, blk in enumerate(model.bottleneck):
            print(f"\n  Block {bi}:")
            conv_out = blk.conv_block(e4)
            stat(f"  conv_out", conv_out)
            
            x_n = blk.norm(e4.permute(0,2,3,1)).permute(0,3,1,2)
            stat(f"  x_normed (into SS2D)", x_n)
            
            with torch.cuda.amp.autocast(enabled=False):
                pos, mom, eng = blk.ss2d(x_n.float())
            
            stat(f"  pos Re(z)", pos)
            stat(f"  mom Im(z)", mom)
            stat(f"  energy H=0.5*(q^2+p^2)", eng)
            
            ham_out = blk.pos_proj(pos)
            stat(f"  ham_out (projected)", ham_out)
            
            g = blk.gate(torch.cat([conv_out.float(), ham_out], 1))
            stat(f"  gate g", g)
            
            g_spatial = g.mean(dim=1)
            stat(f"  gate spatial (ch-avg)", g_spatial)
            
            fused = conv_out.float() * g + ham_out * (1 - g)
            stat(f"  fused output", fused)
            
            conv_contrib = (conv_out.float() * g).abs().mean().item()
            ham_contrib = (ham_out * (1 - g)).abs().mean().item()
            total = conv_contrib + ham_contrib + 1e-8
            print(f"    ConvNeXt contribution: {100*conv_contrib/total:.1f}%")
            print(f"    Hamiltonian contribution: {100*ham_contrib/total:.1f}%")
            
            # Energy map with learned channel attention (v3)
            energy_f = eng.to(e4.dtype)
            ch_w = blk.energy_attn(energy_f)
            ch_w_expand = ch_w.unsqueeze(-1).unsqueeze(-1)
            energy_map = (energy_f * ch_w_expand).mean(dim=1, keepdim=True)
            stat(f"  energy_map (ch-attn)", energy_map)
            
            e4 = fused.to(e4.dtype)
            last_pos, last_mom, last_eng = pos, mom, eng
            last_energy_map = energy_map
        
        # Check individual scan directions
        print("\n--- SS2D scan direction analysis (last block) ---")
        blk = model.bottleneck[-1]
        x_n = blk.norm(e4.permute(0,2,3,1)).permute(0,3,1,2)
        with torch.cuda.amp.autocast(enabled=False):
            x_f = x_n.float()
            for d in range(4):
                lines, h, w = blk.ss2d._to_lines(x_f, d)
                q, p, e = blk.ss2d.scans[d](lines)
                print(f"  Dir {d}: q range=[{q.min():.3f}, {q.max():.3f}]  "
                      f"p range=[{p.min():.3f}, {p.max():.3f}]  "
                      f"e range=[{e.min():.3f}, {e.max():.3f}]")
        
        # Check oscillator parameters
        print("\n--- Oscillator learned parameters ---")
        for d in range(4):
            scan = blk.ss2d.scans[d]
            log_k = scan.log_k
            omega = torch.exp(log_k / 2.0)
            print(f"  Dir {d} omega (freq): min={omega.min():.4f} max={omega.max():.4f} mean={omega.mean():.4f}")
            print(f"  Dir {d} nu_scale: min={scan.nu_scale.min():.4f} max={scan.nu_scale.max():.4f}")
            print(f"  Dir {d} dt_scale: min={scan.dt_scale.min():.4f} max={scan.dt_scale.max():.4f}")
        
        # Energy-gated skip analysis
        # Energy is 1-channel, natural range. Skip gate uses centering: σ(γ*(H - mean(H)))
        print("\n--- Energy-gated skip connections ---")
        energy = last_energy_map  # (B, 1, 28, 28)
        
        for name, skip, enc in [
            ("skip3", model.skip3, e3),
            ("skip2", model.skip2, e2),
            ("skip1", model.skip1, e1),
        ]:
            gamma = skip.energy_gamma.item()
            en_interp = F.interpolate(energy, enc.shape[2:], mode='bilinear', align_corners=False)
            e_centered = en_interp - en_interp.mean(dim=(2, 3), keepdim=True)
            gate_val = torch.sigmoid(gamma * e_centered)
            print(f"  {name}: gamma={gamma:.4f}  "
                  f"energy=[{en_interp.min():.4f}, {en_interp.max():.4f}]  "
                  f"centered=[{e_centered.min():.4f}, {e_centered.max():.4f}]  "
                  f"gate=[{gate_val.min():.4f}, {gate_val.max():.4f}]  "
                  f"gate_mean={gate_val.mean():.4f}")
        
        # Phase-space attention check (v3: centered energy, not energy_proj)
        print("\n--- Phase-space attention ---")
        d3_up = model.up3(e4)
        en3 = F.interpolate(energy, d3_up.shape[2:], mode='bilinear', align_corners=False)
        mom3 = F.interpolate(last_mom, d3_up.shape[2:], mode='bilinear', align_corners=False)
        mom3 = mom3[:, :d3_up.shape[1]]
        
        # v3: centered energy attention (same as skip gates)
        ps_gamma = model.ps_attn.energy_gamma.item()
        e_centered_ps = en3 - en3.mean(dim=(2, 3), keepdim=True)
        alpha = torch.sigmoid(ps_gamma * e_centered_ps)
        print(f"  PS energy_gamma: {ps_gamma:.4f}")
        stat("  alpha (centered energy attn)", alpha)
        
        mom_feat = model.ps_attn.momentum_proj(mom3)
        stat("  momentum projected", mom_feat)
        
        print("\n--- Summary ---")
        print(f"  Gate g mean: {g.mean():.4f} (1.0=pure ConvNeXt, 0.0=pure Hamiltonian)")
        print(f"  Gate spatial std: {g_spatial.std():.6f}")
        print(f"  Position range: [{last_pos.min():.4f}, {last_pos.max():.4f}]")
        print(f"  Momentum range: [{last_mom.min():.4f}, {last_mom.max():.4f}]")
        print(f"  Energy range: [{last_eng.min():.4f}, {last_eng.max():.4f}]")
        print(f"  Energy spatial variance: {last_energy_map.var(dim=(2,3)).mean():.6f}")
        print(f"  Momentum spatial variance: {last_mom.mean(dim=1).var(dim=(1,2)).mean():.6f}")
        print(f"  Position spatial variance: {last_pos.mean(dim=1).var(dim=(1,2)).mean():.6f}")
        
        # Boundary correlation analysis
        msk_small = F.interpolate(msk.unsqueeze(0).unsqueeze(0), size=(28,28), mode='nearest')[0,0]
        from torch.nn.functional import max_pool2d
        msk_s = msk_small.unsqueeze(0).unsqueeze(0)
        dilated = max_pool2d(msk_s, 3, stride=1, padding=1)
        eroded = 1 - max_pool2d(1-msk_s, 3, stride=1, padding=1)
        boundary = ((dilated - eroded) > 0.5).float()[0,0]
        interior = (eroded > 0.5).float()[0,0]
        exterior = ((1 - dilated) > 0.5).float()[0,0]
        
        energy_map = last_energy_map[0, 0].cpu()
        mom_map = last_mom[0].norm(dim=0).cpu()
        
        if boundary.sum() > 0:
            e_boundary = energy_map[boundary > 0.5].mean().item()
            e_interior = energy_map[interior > 0.5].mean().item() if interior.sum() > 0 else 0
            e_exterior = energy_map[exterior > 0.5].mean().item() if exterior.sum() > 0 else 0
            print(f"\n  Energy at boundary:  {e_boundary:.4f}")
            print(f"  Energy at interior:  {e_interior:.4f}")
            print(f"  Energy at exterior:  {e_exterior:.4f}")
            print(f"  Boundary/Interior ratio: {e_boundary/(e_interior+1e-8):.2f}")
            print(f"  Boundary/Exterior ratio: {e_boundary/(e_exterior+1e-8):.2f}")
            
            m_boundary = mom_map[boundary > 0.5].mean().item()
            m_interior = mom_map[interior > 0.5].mean().item() if interior.sum() > 0 else 0
            m_exterior = mom_map[exterior > 0.5].mean().item() if exterior.sum() > 0 else 0
            print(f"\n  Momentum at boundary: {m_boundary:.4f}")
            print(f"  Momentum at interior: {m_interior:.4f}")
            print(f"  Momentum at exterior: {m_exterior:.4f}")
            print(f"  Boundary/Interior ratio: {m_boundary/(m_interior+1e-8):.2f}")
            print(f"  Boundary/Exterior ratio: {m_boundary/(m_exterior+1e-8):.2f}")

if __name__ == '__main__':
    main()
