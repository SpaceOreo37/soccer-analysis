# Licensing research (Phase 0)

> Status: **no license chosen yet.** This page collects the facts so the decision can be made.
> Researched 2026-11-25. This is a summary for decision-making, not legal advice. License
> terms change, so re-check the sources before shipping anything public (especially Phase 5).

## Quick vocabulary

- **Permissive** licenses (MIT, BSD, Apache-2.0): "do almost anything, just keep our copyright
  notice." You can combine them with any license, including a closed-source product.
  Apache-2.0 also includes an explicit patent grant; MIT/BSD are shorter and simpler.
- **Copyleft** licenses (GPL, AGPL): "if you distribute software built on this, the whole
  thing must be released under the same license, with source code."
- **AGPL-3.0** is the strongest common copyleft. It closes the "web app loophole": under the
  plain GPL, running modified software on your own server and letting people use it over the
  internet doesn't count as distributing it, so you'd never have to share source. AGPL section
  13 says that if users interact with a modified AGPL program over a network, you must offer
  them the complete source code of that program, including the code you combined with it.
- **Weak copyleft** (MPL-2.0): only the specific files you modify must stay open. It doesn't
  spread to your code.

## What the pipeline depends on

| Component | Used for | License | Effect on our project |
|---|---|---|---|
| [roboflow/sports](https://github.com/roboflow/sports) | Pitch config, radar drawing, team classifier, example code | **MIT** | None beyond keeping their copyright notice if we copy code. |
| [supervision](https://github.com/roboflow/supervision) | Video I/O, annotators, ByteTrack | **MIT** | None. |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (`ultralytics`, `ultralytics-thop`, `ultralytics-platform`) | Runs the player/ball/pitch models | **AGPL-3.0** | **The deciding factor, see below.** |
| Roboflow's 3 sample models (`football-*-detection.pt`) | Player, ball, pitch keypoint detection | Not stated. Trained with Ultralytics YOLOv8, and Ultralytics says [models trained with its software are AGPL-3.0](https://ultralytics.com/license). | Treat them as AGPL. Never redistribute them (they're already gitignored). |
| Training data behind those models | (Roboflow Universe datasets derived from the [DFL Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) on Kaggle) | Universe forks list CC BY 4.0. The original Kaggle competition data rules weren't confirmed; DFL challenge data is often restricted to the competition. | Fine for learning and experiments. For a public product, retrain on data with clear terms. |
| Sample videos (`2e57b9_0.mp4` etc.) | Testing | Bundesliga footage from the same Kaggle data | Don't redistribute (already gitignored). |
| [SigLIP `google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224) | Team classification embeddings | **Apache-2.0** | None. |
| [RF-DETR](https://github.com/roboflow/rf-detr) (`rfdetr`, not used yet) | Possible YOLO replacement | Package and the Nano–Large models: **Apache-2.0**. XL/2XL detection models (and possibly Atto/Femto/Pico) use Roboflow's **PML-1.0** "Platform Model License", which limits self-hosting. Roboflow's own pages disagree on exactly which sizes. | Apache sizes are safe. Avoid PML sizes or read `LICENSE.platform` first. |
| [trackers](https://github.com/roboflow/trackers) (not used yet) | Where ByteTrack moved after supervision 0.31 | **Apache-2.0** | None. |
| PyTorch, torchvision | Deep learning runtime | BSD-3 / BSD-style | None. |
| transformers, tokenizers, safetensors, huggingface_hub, sentencepiece | Loading SigLIP | Apache-2.0 | None. |
| OpenCV (`opencv-python`) | Image processing, homography | Apache-2.0 | None. |
| scikit-learn, umap-learn, numpy, scipy, numba | Clustering teams, maths | BSD | None. |
| tqdm, certifi | Progress bars, SSL certificates | MPL-2.0 (+MIT) | None unless we edit their files. |
| gdown | Downloading models | MIT | None. |
| ffmpeg (Homebrew build) | Trimming and re-encoding video | **GPL** (built with `--enable-gpl --enable-libx264`) | We run it as a separate program, which doesn't affect our license. It only matters if we ever *bundle* ffmpeg binaries (e.g. in a Docker image); then we'd follow the GPL for ffmpeg itself. |
| [SoccerTrack v2](https://atomscott.github.io/SoccerTrack-v2/) (Phase 1) | Evaluation data | Data **CC BY 4.0**, code MIT. The Hugging Face copy requires accepting dataset terms. | Must credit the authors. Read the HF terms before downloading. |

Everything except Ultralytics (and the models trained with it) is permissive. **So the
license decision is really one question: do we keep Ultralytics YOLO in the pipeline?**

## What AGPL Ultralytics means for this project

- Using it to learn, run experiments, or process your own team's videos on your own computers:
  **no obligations**.
- Publishing the repo openly with Ultralytics as a dependency: allowed. Your code can import
  it.
- **Phase 5 (a web app coaches use over the internet)** is where AGPL section 13 bites. If the
  hosted pipeline uses Ultralytics or YOLO weights, the whole service (backend, worker,
  pipeline) must be offered as source code under AGPL-compatible terms. Ultralytics' own view
  is that even custom-trained weights and the inference code that loads them are covered.
  Since you plan to be open source, that may be fine, but it decides what license the
  *combined* app effectively has.
- Your own files can technically still carry MIT (MIT is compatible with AGPL), but anyone
  receiving the combined app receives it under AGPL terms. In practice "MIT + Ultralytics"
  behaves like AGPL for the app as a whole.
- The paid way out is an [Ultralytics Enterprise License](https://www.ultralytics.com/license),
  meant for closed-source products. It's probably irrelevant for an open-source student project.

## Options

### Option A: AGPL-3.0 for the whole project, keep Ultralytics
- **Meaning:** Anyone can use, change and host the app, but anyone who hosts a modified version
  must publish their changes. Nobody can take your work and turn it into a closed commercial
  product.
- **Pros:** Matches the dependency exactly, so no conflict ever. Zero extra engineering. Keeps
  the project open by force.
- **Cons:** Many companies and some schools/clubs are wary of AGPL, so fewer outside
  contributors or adopters. If you later want a closed-source or commercial version, you'd
  need to remove Ultralytics *and* own (or get permission for) all contributed code.

### Option B: MIT or Apache-2.0, and replace Ultralytics before Phase 5
- **Meaning:** A maximally reusable project. Anyone (including companies) can use it for
  anything with attribution.
- **How:** Keep Ultralytics only in the throwaway Phase 0 `baseline` dependency group (already
  separated in `pyproject.toml`). Never import it from `src/`. In Phase 1–2, switch detection to
  **RF-DETR Apache-licensed sizes**, tracking to **`trackers` (Apache-2.0)**, and train our own
  player/ball/pitch models on data with clear terms (SoccerTrack v2, coach-consented footage).
- **Pros:** Widest adoption, no copyleft questions for the web app, keeps commercial doors
  open.
- **Cons:** Real extra work: training three models (a good job for the RX 7900 XTX) and
  rewriting the example's model calls. Someone could build a closed product on your work.
- **MIT vs Apache-2.0:** Apache adds an explicit patent license and a "state your changes"
  rule. It's a bit more protective, a bit longer. Both are fine.

### Option C: Permissive now, decide about Ultralytics later
- License your own code MIT/Apache now, and keep using Ultralytics during research (Phases 1–4).
  Before the web app goes live, either switch to Option B's stack or relicense to AGPL
  (Option A).
- **Pros:** Doesn't block progress today.
- **Cons:** Contributors will have submitted code under MIT/Apache. Moving to AGPL later is
  allowed (permissive code can be included in an AGPL work), but moving *away* from AGPL later
  would need all contributors' agreement. Requires discipline to keep YOLO out of `src/`.

### Option D: Closed source + Ultralytics Enterprise License
- Listed for completeness. It conflicts with your open-source goal and costs money.

## Not licensing, but related (for Phase 6)

Footage of youth/high-school players is personal data of minors. Before piloting with real
coaches, plan for consent (players/parents), who can see the videos, how long they're
kept, and never committing real match footage anywhere public (the repo already blocks videos).

## Sources
- Ultralytics licensing: https://ultralytics.com/license, https://github.com/ultralytics/ultralytics/issues/2129
- Roboflow on AGPL in CV: https://roboflow.com/blog/why-agpl-3-0-is-a-risk-for-computer-vision-teams
- RF-DETR licensing: https://roboflow.com/model-licenses/rf-detr, https://blog.roboflow.com/rf-detr-is-free-to-use-commercially/, https://github.com/roboflow/rf-detr
- SigLIP model card: https://huggingface.co/google/siglip-base-patch16-224
- SoccerTrack v2: https://atomscott.github.io/SoccerTrack-v2/, https://arxiv.org/abs/2508.01802
- AGPL section 13 explainers: https://choosealicense.com/licenses/agpl-3.0/, https://opensource.com/article/17/1/providing-corresponding-source-agplv3-license
- Dataset: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
- Package licenses: read from installed package metadata (`importlib.metadata`) and PyPI.
