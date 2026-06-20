# L08 — Computer Vision & Convolutional Neural Networks

> **Module 3 · Machine Learning & GenAI · Lesson 8 of 10**

## The story so far

Sarah's neural network model from L07 ships fine for predicting checkout completion — and on tabular session data, a logistic regression with good features performs just as well. The NorthStar product team is unblocked.

Then Marcus drops the next request: *"Our merchandising team uploads thousands of new product photos every season. Can we auto-tag them as 'dress', 'shirt', 'sneaker' so the catalogue search works on day one?"*

This is a different problem. The input isn't 9 numeric features anymore — it's a 28×28 (or larger) grid of pixels. A flat MLP on flattened pixels works ~87% of the time, which sounds OK until you realise that means roughly 1 in 8 product images gets mis-tagged.

This is where **convolutional neural networks** earn their reputation.

## What you'll learn

By the end of L08 you will be able to:

1. **Reason about image data** as tensors `(batch, channels, height, width)` and explain why flattened MLPs struggle with images.
2. **Apply convolutions** by hand — pick a kernel (edge detector, blur) and predict the output feature map.
3. **Build a CNN in PyTorch** from `Conv2d` + `MaxPool2d` + `Linear` blocks, trained end-to-end on Fashion-MNIST.
4. **Use transfer learning** — start from a pretrained model, freeze the backbone, fine-tune the head. Compare against training from scratch on small data.
5. **Recommend the right tool** for a real product imaging problem: when is a tiny custom CNN enough, when do you reach for a pretrained backbone, and when do you just call an API?

## Phase 1 — Pre-class (≈ 30 min)

- Watch the [lesson intro video](https://youtu.be/WasU-87LgfA) (~5 min)
- Read [pre-class.md](pre-class.md)
- Watch the linked 3Blue1Brown convolution video
- Run [notebooks/01_monday_morning.ipynb](notebooks/01_monday_morning.ipynb) — flatten Fashion-MNIST, train an MLP, and feel the parameter explosion

## Phase 2 — In-class (≈ 90 min lecture + 90 min code-along)

- Concept walkthrough: instructor uses the [**interactive key-concepts walkthrough →**](https://su-ntu-ctp.github.io/6m-data-3.8-Computer-Vision/) (revisit any time)
- Code-along notebooks (in order):
  - [02_convolutions_intuition.ipynb](notebooks/02_convolutions_intuition.ipynb) — manual kernels & feature maps
  - [03_first_cnn.ipynb](notebooks/03_first_cnn.ipynb) — a tiny CNN that beats the MLP
  - [04_transfer_learning.ipynb](notebooks/04_transfer_learning.ipynb) — pretrained ResNet18 on small data

> **Don't have a GPU?** L08 runs end-to-end on CPU (~30 min for the CIFAR + ResNet18 fine-tune), but on a T4 GPU in Colab it's ~2 min. Each notebook has an **Open in Colab** badge at the top — click it, then **Runtime → Change runtime type → T4 GPU**:

## Phase 3 — Post-class (self-study, optional)

- [assignment.ipynb](notebooks/assignment.ipynb) — CIFAR-10 classifier: CNN from scratch vs transfer learning
- [optional_extensions.ipynb](notebooks/optional_extensions.ipynb) — data augmentation, BatchNorm, learning-rate schedulers

## Files

| Path | Purpose |
|------|---------|
| `README.md`              | This file — orientation |
| `setup.md`               | Environment install (`pip install torch torchvision`) |
| `pre-class.md`           | ~30-min self-study before class |
| `lesson.md`              | Short reference: overview, takeaways, transfer-learning checklist, review Q&A, L09→L10 course map |
| `reference.md`           | Glossary of 20 CV/CNN terms |
| `environment.yml`        | Conda env spec |
| `docs/index.html`        | Interactive key-concepts walkthrough (GitHub Pages) |
| `notebooks/`             | 4 in-class NBs + assignment + extensions + data |

---

**Bridge to L09:** Once we can tag images, the catalogue team asks: *"Can we also do natural-language search? A customer types 'blue summer dress' — match it to the right product."* That's text, not pixels. Different modality, same neural-network playbook. See L09.
