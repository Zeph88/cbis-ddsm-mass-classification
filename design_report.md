<h1>Overview of the project</h1>

According to World Health Organisation, breast cancer is one of the leading cause of deaths among women. It was the most common cancer in women in 157 countries out of 185 in 2022. 670,000 women died from this desease worldwide in the same year. Statistically, over the course of their lives, 1 woman out of 12 will suffer from breast cancer, making a compelling case for prevention. Although the techniques may vary, one of the most prevalent is mammography. It is a well-established imaging technique that has undergone significant improvements ranging from digital mammography to digital breast tomosynthesis [1]. Historically, the imagery was investigated by specialists only, but the surge of CNN-based methods assist the diagnostic from now on.

Relating to the template 3.2. "Deep Learning Breast Cancer Detection", the project aims to classify suspicious mammographic abnormalities as benign or malignant. Mammography review can indeed be broken down into two smaller steps. Identifying whether there is an abnormality displayed on the imagery (CADe) ; assessing whether this abnomality is an aggressive form (CADx). The emphasize is put on the latter.

<h1>Domain and users</h1>

As previously hinted, this project consists in building a pipeline for breast cancer diagnostic. Diagnostic (CADx) differs from detection (CADe). In such case, it is acknowledged the patient has an abnormality. However, the subsequent decisions with regard to the candidate treatment depend on the nature of the abnormality, hence the importance of assessing whether it is a benign or a malignant form.

From this perspective, the user and the target population should be differentiated. Assessing the nature of a tumor is a professional radiologist's job. Such a model would support radiologists by providing a second-opinion signal, helping prioritise uncertain cases, and flagging predictions that require closer review. His judgment remains decisive. However, the pipeline could assist with triage while arbitrating edge cases. The distinction across both situations is the level of confidence of the model. As a result, such pipeline needs to display transparency and explainability to point towards cases that require further examination. Rezazade Mehrizi, M.H., Mol, F., Peter, M. et al. emphasizes the importance of decision system evaluation. AI suggestions can affect radiologists’ decisions in mammography examination, making explainability and careful human-AI interaction central to clinical deployment. Indeed, when they do not spend extensive time investigating a mammogram, radiologists are prone to approve an incorrect decision made by AI. Some explainability tools mitigate this bias, especially visual heat maps [2].

Although breast cancer can also occur in men, male cases account for only around 0.5–1% of breast cancer cases. The target population represented by mammography screening datasets is therefore predominantly female.

<h1>Design trade-offs</h1>

<h2>Dataset</h2>

The model is trained and tested on the CBIS-DDSM dataset. It is a curated version of the DDSM dataset which cut part of the noise introduced by unreliable image resolution and mislabelling in the original dataset [3]. Every case contains a few DICOM files ("digital imaging and communications in medicine").

    - A cropped image focused on the abnormality
    - A full picture of the breast that includes the abnomality
    - A binary-mask file that can be applied to the full picture of the breast to locate the tumor 

The aforementioned three files are supposed to be available twice for each case, one for the CC view, one for the MLO view. Varoquaux and Cheplygina stresses that patient leakage may inflate the performance of the model when tested [4]. Although the train and the test sets do not overlap, the split between train and validation sets may, because it is not part of the CBIS-DDSM specifics. To mitigate the risk of leakage, the project relies on a fixed split, done at patient level, kept in three csv files - train_split, val_split, test_split - to ensure the experiments are carried based on similar conditions and can be reproduced. The train / validation split is made based on a basic 80% / 20% ratio of the train set.

<h2>Data pre-processing</h2>

DICOM files are raw documents at this stage ; they need to be preprocessed to some extent to help the model learns efficiently. However, the preprocessing phase already raises important questions. Should the tumour be used as input alone or some context be associated to it ? Should this context be blurred to orient the model toward the region of interest ("ROI") or should it be left vanilla ?

The literature is not one-sided on this important matter. As a prerequisite, this project does not consider the default cropped image as a candidate for training. There is no control over the margin around the tumour and, as a result, no possibility to normalize the input. It is used for the comparison, but as a benchmark to assess whether the preprocessing helps or not. Trading off the preprocessing dilemma is indeed a matter of experiment. A simple baseline model was built that has three layers of 32, 64 and 128. To ensure a fair comparison, every DICOM is converted to a 224 x 224 pixel image. For every preprocessing decision, three seeds (42, 123 and 999) have been tested to prevent that an exceptional run on one format biases the conclusion. The results are disclosed in Table 1 (##Table-1).

The emphasis factor stresses the abnormality while the soft mask darkens the context. The margin is expressed in pixels. From these experiments, it can be inferred the context is critical to keep. Slightly blurring it also improves the results. 

<h2>Architecture</h2>

The preliminary experiments suggest that the way contextual information is handled has a measurable effect on model performance. In particular, zooming aggressively around the tumour appears to reduce useful anatomical context, whereas soft masking can guide the model toward the ROI while preserving surrounding breast tissue. This supports a two-level design in which global context and local lesion detail are treated as complementary sources of information.

This design is inspired by Shen et al.’s GMIC architecture, which combines a global module operating on high-resolution mammograms with a local module focused on selected informative regions [5]. However, the present project is narrower in scope: it focuses on CADx classification of known abnormalities rather than full CADe localisation. The proposed pipeline therefore trains two complementary branches: a global branch using the full mammogram representation, and a local branch using 256×256 ROI-centred patches. Their features will then be merged before the final benign/malignant classification layer.

Based on the preprocessing experiments, soft masking will be applied to guide both local and global inputs, while avoiding complete removal of contextual tissue. The objective is to preserve clinically relevant anatomical context while giving the model stronger access to lesion-specific information.

<p>The resulting architecture follows a two-branch structure:</p>

<pre>
Full mammogram + soft mask
        → global CNN branch
        → global feature vector

ROI-centred 256×256 patch + soft mask
        → local CNN branch
        → local feature vector

global features + local features
        → feature fusion
        → dense classification layers
        → benign / malignant prediction
</pre>


<h1>Work plan</h1>

The project will be developed in four main phases: data preparation, representation selection, final modelling, and evaluation/reporting. The work plan is deliberately staged so that a robust baseline and preprocessing comparison can be delivered even if the more ambitious global-local architecture cannot be fully completed.

| Task | June | July | August | September |
|---|:---:|:---:|:---:|:---:|
| Finalise dataset ingestion, DICOM path resolution and metadata matching | ███ |  |  |  |
| Validate ROI-mask alignment and preprocessing outputs through visual checks | ███ |  |  |  |
| Complete preprocessing comparison using fixed train/validation/test splits | ███ | █ |  |  |
| Select best image representation based on AUC, recall and stability across seeds |  | █ |  |  |
| Train improved single-branch CNN on selected representation |  | ███ | █ |  |
| Implement simplified global-local model, if feasible |  | ██ | ██ |  |
| Tune regularisation, augmentation and threshold selection |  |  | ███ |  |
| Generate evaluation tables, error analysis and interpretability outputs |  |  | ██ | ██ |
| Write final report and prepare figures / appendix |  |  | █ | ███ |


https://www.who.int/news-room/fact-sheets/detail/breast-cancer

[1] Wang L (2024) Mammography with deep learning for breast cancer detection. Front. Oncol. 14:1281922. doi: 10.3389/fonc.2024.1281922

[2] Rezazade Mehrizi, M.H., Mol, F., Peter, M. et al. The impact of AI suggestions on radiologists’ decisions: a pilot study of explainability and attitudinal priming interventions in mammography examination. Sci Rep 13, 9230 (2023). https://doi.org/10.1038/s41598-023-36435-3

[3] Lee, R., Gimenez, F., Hoogi, A. et al. A curated mammography data set for use in computer-aided detection and diagnosis research. Sci Data 4, 170177 (2017). https://doi.org/10.1038/sdata.2017.177

[4] Varoquaux, G., Cheplygina, V. Machine learning for medical imaging: methodological failures and recommendations for the future. npj Digit. Med. 5, 48 (2022). https://doi.org/10.1038/s41746-022-00592-y

[5] Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu, Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras, An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization, Medical Image Analysis, Volume 68, 2021, 101908, ISSN 1361-8415, https://doi.org/10.1016/j.media.2020.101908.

<h1>Appendix</h1>

## Table 1

| Experiment | Input strategy | Seed 42 AUC | Seed 123 AUC | Seed 999 AUC | Mean AUC | Best observed result | Interpretation |
|---|---|---:|---:|---:|---:|---|---|
| Baseline cropped | Curated cropped images, 224×224 | 0.653 | 0.639 | 0.638 | 0.643 | Seed 42: AUC 0.653; val AUC 0.607 | Reasonable baseline, but performance remains modest and unstable across thresholds. |
| Soft mask + zoom | Soft factor 0.3, zoom margin 30 | 0.631 | 0.623 | 0.628 | 0.627 | Seed 999: AUC 0.628; val AUC 0.604 | Slightly worse than the cropped baseline. Zooming around the ROI does not clearly improve performance. |
| Soft mask + zoom | Soft factor 0.7, zoom margin 30 | 0.612 | 0.587 | 0.603 | 0.601 | Seed 42: val AUC 0.623; Seed 123: val AUC 0.620 | Increasing the soft factor with zoom appears to degrade performance. |
| Emphasis + zoom | Emphasis factor 0.25, zoom margin 30 | 0.572 | 0.566 | 0.568 | 0.569 | Seed 42: val AUC 0.630 | Weak test performance. Emphasising the ROI while zooming seems to remove useful contextual information. |
| Emphasis + zoom | Emphasis factor 0.5, zoom margin 30 | 0.546 | 0.572 | 0.584 | 0.567 | Seed 123: val AUC 0.606; Seed 999: val AUC 0.629 | Poor overall. Stronger emphasis does not help and may make predictions unstable or biased toward one class. |
| No mask, no zoom | Full image context, 224×224 | 0.664 | 0.601 | 0.686 | 0.650 | Seed 999: AUC 0.686; accuracy 0.635 at threshold 0.5 | Keeping contextual information performs better than most zoomed ROI variants. |
| Soft mask, no zoom | Soft factor 0.3, no zoom | 0.687 | 0.735 | 0.715 | 0.712 | Seed 123: AUC 0.735; accuracy 0.690 at threshold 0.45 | Best complete configuration. Soft ROI information helps when global image context is preserved. |
| Soft mask, no zoom | Soft factor 0.7, no zoom | N/A | 0.733 | N/A | N/A | Seed 123: AUC 0.733; val AUC 0.655 | Very promising, but incomplete because only seed 123 is available. Needs seeds 42 and 999 before conclusion. |