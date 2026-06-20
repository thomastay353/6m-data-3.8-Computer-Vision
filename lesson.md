# Lesson — L08 Computer Vision & Convolutional Neural Networks

> **Chapter 8 of the NorthStar Retail story.** *Sarah Chen · Customer Experience Analyst · Mid-year, post-L07.*
> Sarah's L07 neural network for checkout-completion has shipped. Then Marcus walks over: *"Our merchandising team uploads thousands of new product photos every season. Can we auto-tag them as 'dress', 'shirt', 'sneaker' so the catalogue search works on day one?"*
> Same network playbook from L07 — but pixels are not tabular features. This lesson is how Sarah rebuilds her toolkit for images.

This document is a **short reference** — the lesson itself is taught in the notebooks. Read it for orientation before class, then come back to it for the takeaways, the transfer-learning checklist, the review questions, and the course map.

---

## How L08 is taught

| Stage | Where to go |
|---|---|
| **Pre-class** | `pre-class.md` + `notebooks/01_monday_morning.ipynb` |
| **In-class — Part 1: Convolutions intuition** | `notebooks/02_convolutions_intuition.ipynb` |
| **In-class — Part 2: Build your first CNN** | `notebooks/03_first_cnn.ipynb` |
| **In-class — Part 3: Transfer learning** | `notebooks/04_transfer_learning.ipynb` |
| **Self-study** | `notebooks/assignment.ipynb` + `notebooks/optional_extensions.ipynb` |
| **Reference & review** | This document |

The notebooks are the spine. Run them in order. Come back here for the consolidated takeaways and the review questions.

---

## Overview

Sarah's L07 MLP solved a tabular problem (9 numeric features → checkout-completion probability). Marcus's catalogue-tagger problem looks superficially similar — "another classifier" — but the input is a grid of pixels, and a flattened MLP on 224×224 colour photos blows up to ~38 million parameters in the first layer alone. Worse, the MLP has no built-in notion that pixel (10, 10) sits next to pixel (10, 11) — it has to *learn* spatial structure from scratch. **Convolutional neural networks (CNNs)** fix both problems by sliding a small learnable kernel over the image: locality and translation invariance baked in, parameter count slashed by orders of magnitude. And when Sarah's actual data is 500 photos per category rather than 60,000, she'll reach for **transfer learning** — starting from a ResNet18 someone already trained on 1.2M ImageNet photos — and decide whether head-only feature extraction is enough or whether the domain gap demands fine-tuning the last conv block too.

---

## Key takeaways

1. **Images are tensors, but flattening them is wasteful.** PyTorch's convention is `(batch, channels, height, width)`. A flat-MLP on a 224×224 colour photo needs ~38M parameters in the first layer alone and has to relearn spatial structure from scratch.
2. **Convolutions exploit locality and translation invariance.** A single 3×3 kernel — 9 weights — slides across the whole image. The same edge-detector finds an edge at the top-left and the bottom-right; the MLP would need separate weights for each location.
3. **The classic CNN block is `Conv → ReLU → MaxPool`.** Stride controls down-sampling; padding preserves output size; pooling discards precise location and keeps the strongest activation. Stack a few blocks and the network learns edges → parts → objects.
4. **The PyTorch training loop is unchanged from L07.** `Conv2d` and `MaxPool2d` are drop-in replacements for `Linear`. The `zero_grad → forward → loss → backward → step` pattern still works — CNNs are a different *architecture*, not a different framework.
5. **Don't train from scratch on small data — start pretrained.** ResNet/EfficientNet/ViT weights from ImageNet encode a hierarchy of visual features (edges, textures, parts, objects) that transfer to most natural-photo tasks. With < 1K images per class, transfer learning is almost always the right starting point.
6. **Feature extraction vs fine-tuning depends on the domain gap.** Pretrained domain close to yours (ImageNet → product photos) → freeze the backbone, train only the new head. Domain far away (ImageNet → grayscale silhouettes, X-rays, satellite) → unfreeze the last conv block and fine-tune with a tiny learning rate (e.g. `1e-4`). Head-only transfer can actually *underperform* a from-scratch CNN on a poorly-matched domain.
7. **Pretrained models impose preprocessing contracts.** Match ImageNet normalisation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), resize inputs to a sensible scale for the backbone's receptive field, and convert grayscale to RGB if needed. Mismatched preprocessing silently destroys accuracy.

---

## Choosing a transfer-learning strategy — a checklist

Before you decide whether to train from scratch, extract features, or fine-tune, run the project through this three-step check:

1. **How much labelled data do I actually have, per class?** Under ~1K per class → start pretrained, do not train from scratch. 10K+ per class → from-scratch CNN becomes viable; pretrained is still a strong baseline to beat.
2. **How close is my domain to ImageNet?** Colour natural-photo inputs (product photos, animals, vehicles, food) → head-only feature extraction will already lift you a lot. Grayscale, medical scans, satellite, line drawings → expect head-only to underperform; budget for fine-tuning the last conv block (`layer4` in ResNet).
3. **Have I matched the pretrained model's preprocessing contract?** ImageNet normalisation stats, RGB (3-channel) inputs, an input resolution the backbone was designed for. If you skip this, your "transfer-learning baseline" is measuring preprocessing bugs, not the model.

Skip any of these and you will reach for fine-tuning when feature extraction would suffice, or report a from-scratch CNN as "better than transfer learning" when really your preprocessing was wrong.

---

## Check your understanding

Work through these after finishing the three Part notebooks. Attempt each question on your own first.

### Part 1 — Convolutions intuition

**Q1 — Why not just flatten?** A teammate says: *"Let's just flatten the 224×224 colour photo and feed it to the same MLP that worked for L07."* Give two distinct reasons why this is a poor choice.

> **Sample answer:** (1) **Parameter count.** Flattened input is 3 × 224 × 224 = 150,528 features. Even a modest 256-neuron hidden layer needs ~38 million weights just in that one layer — untenable to train and memorise. (2) **No spatial prior.** An MLP treats every pixel as independent of every other pixel. It has no built-in notion that pixel (10, 10) sits next to pixel (10, 11), and no way to recognise that a sleeve in the top-left is the same feature as a sleeve in the bottom-right. The model has to learn locality and translation invariance from scratch, from training data — expensive and fragile.

**Q2 — Kernel arithmetic.** A `Conv2d(in_channels=3, out_channels=32, kernel_size=3)` layer is applied to a 224×224 colour image. How many learnable parameters does the layer have, and how does that compare to the equivalent MLP first layer producing 32 outputs?

> **Sample answer:** Conv parameters = 32 kernels × (3 channels × 3 × 3) + 32 biases = 864 + 32 = **896**. MLP equivalent (Linear(150528 → 32) + bias) = 150,528 × 32 + 32 ≈ **4.82 million**. The convolution is roughly **5,000× smaller** — because the same kernel is reused at every spatial location instead of having separate weights per pixel.

**Q3 — Stride, padding, pooling.** In one sentence each, what does **stride 2**, **padding 1** (with kernel 3), and a **2×2 max-pool** do to the output feature map's spatial size?

> **Sample answer:** **Stride 2** halves the output height and width (the kernel jumps by 2 pixels at a time). **Padding 1** with a 3×3 kernel keeps the output the same size as the input (the border of zeros lets the kernel reach the edges). **2×2 max-pool** halves the height and width and keeps the strongest activation in each 2×2 block.

### Part 2 — Building a CNN

**Q4 — Output-shape tracing.** A `TinyCNN` has `Conv2d(1, 16, 3, padding=1) → MaxPool2d(2) → Conv2d(16, 32, 3, padding=1) → MaxPool2d(2)` applied to a 1×28×28 Fashion-MNIST image. What is the shape going into the first `Linear` layer after `Flatten()`?

> **Sample answer:** After the first conv (padding 1 keeps size): 16×28×28. After the first pool: 16×14×14. After the second conv: 32×14×14. After the second pool: 32×7×7. Flattened: **32 × 7 × 7 = 1,568 features**. The first `Linear` therefore needs `in_features=1568`.

**Q5 — Training-loop reuse.** Your CNN gets ~88% on Fashion-MNIST; the MLP from L07 got ~87%. A colleague says: *"Tiny improvement — and a whole new framework to learn."* Correct them.

> **Sample answer:** The framework is unchanged — `zero_grad → forward → loss → backward → step` is identical to L07. `Conv2d`/`MaxPool2d` are drop-in replacements for `Linear`. The Fashion-MNIST gap looks small (88 vs 87%) because the dataset is easy and small, but it is achieved with about half the parameters; and the CNN's advantage over a flattened MLP grows sharply with image complexity — Fashion-MNIST silhouettes hide it, colour photos like CIFAR-10 (assignment) make it obvious.

### Part 3 — Transfer learning

**Q6 — Strategy choice.** You have 500 product photos per category across 12 categories of clothing — colour, taken in studio. Should you (a) train a CNN from scratch, (b) use a pretrained ResNet18 with head-only feature extraction, or (c) use the pretrained ResNet18 and fine-tune the last conv block? Justify.

> **Sample answer:** **(b) — head-only feature extraction is the right first move, and (c) is a reasonable second move.** 500 images per class is far too small to train a CNN from scratch (it will overfit). The domain — colour studio photos of clothing — is close to ImageNet, so the pretrained backbone's edge/texture/part detectors transfer directly; freezing the backbone and training only the new 12-output head usually lifts accuracy substantially in a few minutes. If you need more, unfreeze `layer4` and fine-tune with a small learning rate (`1e-4`) for a few more epochs. Skip the from-scratch option.

**Q7 — Domain-gap surprise.** On Fashion-MNIST (NB 04) head-only transfer learning **lost** to a from-scratch CNN. On CIFAR-10 (assignment) head-only transfer **wins**. What is the structural difference, and what should you do when transfer learning underperforms?

> **Sample answer:** Fashion-MNIST is **grayscale 28×28 silhouettes**; CIFAR-10 is **colour 32×32 natural photos**. ImageNet pretraining is colour natural photos, so the feature hierarchy transfers much better to CIFAR-10. When the pretrained domain is far from yours (grayscale, X-rays, satellite imagery), head-only feature extraction can underperform because the frozen filters were tuned for the wrong kind of image. The fix is to **unfreeze the last conv block** (e.g. `layer4` in ResNet) and fine-tune with a tiny learning rate (`1e-4`), letting the deepest features adapt to your data while leaving the early generic edge/texture filters intact.

**Q8 — Preprocessing pitfall.** You load a pretrained ResNet18, replace `fc`, train the head on your 64×64 grayscale dataset, and accuracy is dismal. Name three preprocessing checks before you blame the model.

> **Sample answer:** (1) **Channels** — ResNet18 expects 3-channel RGB input. If your data is grayscale, convert to 3-channel (e.g. by repeating the channel) before feeding it in. (2) **Normalisation** — apply ImageNet normalisation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Using your own dataset's stats, or skipping normalisation entirely, silently destroys accuracy because the pretrained weights expect that exact pixel scale. (3) **Input size** — ResNet18 was designed for ~224×224 input. 64×64 is too small for its receptive field; resize up (96×96 minimum, 224×224 if you can afford it) before normalising.

---

## Where L08 fits in the course

L08 closes the supervised-deep-learning arc and hands the network playbook to a different modality.

| Lesson | How L08 shows up |
|---|---|
| **L09 — NLP & Embeddings** | Pretrained-model thinking carries over from vision to text. The "freeze the backbone, swap the head" pattern becomes "use a pretrained encoder, build a downstream task on top of the embeddings." Domain-gap reasoning still applies. |
| **L10 — Transformers & GenAI** | Vision Transformers (ViT) replace the CNN inductive bias with attention, but the transfer-learning playbook (pretrain on giant data, fine-tune for your task) is exactly what L08 trained you for. Multimodal models (CLIP) glue an L08-style image encoder to an L09-style text encoder. |

---

> *"OK, the photo tagger ships. But Marcus has one more ask: can a customer type 'blue summer dress' and we surface the right products? Not from tags — from the words themselves."* — Marcus, after the catalogue launch.
>
> That question — *can we make text-to-product search work?* — is the engine of **L09 (NLP & Embeddings)**.
