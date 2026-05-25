Project DBIS-DDSM

Pipeline:
rely on csv mass_case_description_test_set and mass_case_description_train_set
split mass_case_description_train_set between train et validation set > Done hors pipeline, pour figer les splits
La seed est fixée au début du pipeline pour rendre l'expérience reproductible
Les fichiers csv contiennent les infos sur les chemins ROI et crop image. Ceux-ci sont erronés par rapport aux fichiers dcm enregistrés. > build_dataset.py vise à résoudre les paths. Le module convertit aussi les labels en binary. La difficulté vient de la différenciation entre ROI et crop. Chaque dicom contient des metadata contenant l'info dans SeriesDescription. 
Savoir ou les fichiers se trouvent est insuffisant pour entrainer le CNN. Convertir les dicoms en numpy arrays pendant le processus de training est trop couteux > La conversion en .npy est faite ex-ante. 3 types de conversion sont réalisées : hard mask, soft mask, blended mask. Lors des tests, c'est le soft mask qui semblait donner les meilleurs résultats
J'ai testé plusieurs résolutions. La basique 224 x 224 semble avoir le meilleur ratio couts / bénéfices. Passer en 256 x 256 n'a pas amélioré le modèle.
J'ai testé plusieurs baselines. 
	1. 2 convolutional blocks, filters: 32 et 64. ReLu activation. Dropout final à 0.5. Epoch 10

		minimum probability : 0.33079907298088074, maximum probability : 0.5747036933898926, average probability : 0.43374714255332947
		AUC: 0.5863592190122804
		threshold : 0.35, accuracy : 0.3888888888888889, precision : 0.3882978723404255, recall : 0.9931972789115646
		threshold : 0.4, accuracy : 0.4470899470899471, precision : 0.4049079754601227, recall : 0.8979591836734694 <--- Meilleur recall+
		threshold : 0.45, accuracy : 0.5952380952380952, precision : 0.47, recall : 0.3197278911564626 <--- Meilleur recall / precision trade-off
		threshold : 0.5, accuracy : 0.6296296296296297, precision : 0.7058823529411765, recall : 0.08163265306122448

		seed 123

		minimum probability : 0.36552950739860535, maximum probability : 0.5522816181182861, average probability : 0.4283372163772583
		AUC: 0.594251553435227
		threshold : 0.35, accuracy : 0.3888888888888889, precision : 0.3888888888888889, recall : 1.0
		threshold : 0.4, accuracy : 0.4523809523809524, precision : 0.40625, recall : 0.8843537414965986 <--- Meilleur optimisation recall
		threshold : 0.45, accuracy : 0.6243386243386243, precision : 0.5333333333333333, recall : 0.272108843537415 <--- Meilleur, mais faible recall
		threshold : 0.5, accuracy : 0.6190476190476191, precision : 0.6363636363636364, recall : 0.047619047619047616
		
		pas vraiment d'overfitting : accuracy: 0.5569 vs val_acc : 0.5341

		seed 999

		minimum probability : 0.3281101584434509, maximum probability : 0.5868890881538391, average probability : 0.41227400302886963
		AUC: 0.5819418676561533
		threshold : 0.35, accuracy : 0.3862433862433862, precision : 0.38605898123324395, recall : 0.9795918367346939
		threshold : 0.4, accuracy : 0.5, precision : 0.40869565217391307, recall : 0.6394557823129252
		threshold : 0.45, accuracy : 0.626984126984127, precision : 0.5882352941176471, recall : 0.1360544217687075
		threshold : 0.5, accuracy : 0.6164021164021164, precision : 0.625, recall : 0.034013605442176874

		pas d'overfitting : accuracy: 0.5541 vs val_acc : 0.5915

	Modèle un peu meilleur que le hasard. pas d'overfitting visible. Signal faible. Nécessité d'augmenter la capacité du modèle
	
	2. 3 convolutional blocks, filters: 32, 64 et 128. ReLu activation. Dropout final à 0.5. Epoch 30

		seed 42

		minimum probability : 0.5529760122299194, maximum probability : 0.5769088268280029, average probability : 0.5609666109085083
		AUC: 0.5358541685072298
		threshold : 0.35, accuracy : 0.3888888888888889, precision : 0.3888888888888889, recall : 1.0
		threshold : 0.4, accuracy : 0.3888888888888889, precision : 0.3888888888888889, recall : 1.0
		threshold : 0.45, accuracy : 0.3888888888888889, precision : 0.3888888888888889, recall : 1.0
		threshold : 0.5, accuracy : 0.3888888888888889, precision : 0.3888888888888889, recall : 1.0

		modele n'apprend rien. Tous les cas sont malins par defaut

		seed 123
		
		minimum probability : 0.29527559876441956, maximum probability : 0.7222510576248169, average probability : 0.4480847418308258
		AUC: 0.5997879671349059
		threshold : 0.35, accuracy : 0.3941798941798942, precision : 0.3891891891891892, recall : 0.9795918367346939
		threshold : 0.4, accuracy : 0.4576719576719577, precision : 0.4, recall : 0.7891156462585034
		threshold : 0.45, accuracy : 0.5978835978835979, precision : 0.4825174825174825, recall : 0.46938775510204084
		threshold : 0.5, accuracy : 0.6296296296296297, precision : 0.5507246376811594, recall : 0.2585034013605442
		
		pas d'overfitting : accuracy: 0.5740 vs val_acc : 0.5795

		seed 999

		minimum probability : 0.228410542011261, maximum probability : 0.7722001075744629, average probability : 0.4597647190093994
		AUC: 0.6032335011926848
		threshold : 0.35, accuracy : 0.43386243386243384, precision : 0.40233236151603496, recall : 0.9387755102040817
		threshold : 0.4, accuracy : 0.47619047619047616, precision : 0.4072727272727273, recall : 0.7619047619047619
		threshold : 0.45, accuracy : 0.582010582010582, precision : 0.4696132596685083, recall : 0.5782312925170068
		threshold : 0.5, accuracy : 0.6058201058201058, precision : 0.49, recall : 0.3333333333333333

		pas d'overfitting : accuracy: 0.5778 vs val_acc : 0.5833

	
	Selon la seed 42, le modèle prédit que la tumeur est toujours maligne > collapse du à un modele encore instable.
	bloc avec perf sensiblement identiques à la baseline

	3. 4 convolutional blocks, filters: 32, 64, 128 et 256. ReLu activation. Dropout final à 0.3. Epoch 30

		seed 42
 
		minimum probability : 0.22944091260433197, maximum probability : 0.8156935572624207, average probability : 0.5067823529243469
		AUC: 0.6048531966899313
		threshold : 0.35, accuracy : 0.3915343915343915, precision : 0.38873994638069703, recall : 0.9863945578231292
		threshold : 0.4, accuracy : 0.42063492063492064, precision : 0.39655172413793105, recall : 0.9387755102040817
		threshold : 0.45, accuracy : 0.5079365079365079, precision : 0.42585551330798477, recall : 0.7619047619047619 
		threshold : 0.5, accuracy : 0.5740740740740741, precision : 0.4588235294117647, recall : 0.5306122448979592 <--- Meilleur

		signe d'overfitting : accuracy: 0.63 vs val_acc : 0.6098 ET auc : 0.6739 vs val_auc : 0.6492

		seed 123

		minimum probability : 0.2661837637424469, maximum probability : 0.82634437084198, average probability : 0.5160449743270874
		AUC: 0.6215802338251317
		threshold : 0.35, accuracy : 0.3941798941798942, precision : 0.38858695652173914, recall : 0.9727891156462585
		threshold : 0.4, accuracy : 0.43386243386243384, precision : 0.40175953079178883, recall : 0.9319727891156463
		threshold : 0.45, accuracy : 0.47619047619047616, precision : 0.4117647058823529, recall : 0.8095238095238095
		threshold : 0.5, accuracy : 0.5555555555555556, precision : 0.4472361809045226, recall : 0.6054421768707483

		signe d'overfitting : accuracy: 0.6271 vs val_acc : 0.5947 ET auc : 0.6641 vs val_auc : 0.6376

		seed 999

		minimum probability : 0.17550742626190186, maximum probability : 0.6408672332763672, average probability : 0.3417794108390808
		AUC: 0.6315339988809376
		threshold : 0.35, accuracy : 0.6296296296296297, precision : 0.5299145299145299, recall : 0.4217687074829932
		threshold : 0.4, accuracy : 0.6481481481481481, precision : 0.6521739130434783, recall : 0.20408163265306123
		threshold : 0.45, accuracy : 0.6455026455026455, precision : 0.7241379310344828, recall : 0.14285714285714285
		threshold : 0.5, accuracy : 0.626984126984127, precision : 0.6875, recall : 0.07482993197278912

		signe d'overfitting mais non confirmé sur AUC : accuracy: 0.6176 vs val_acc : 0.553 ET auc : 0.6544 vs val_auc : 0.6494


	4. 4 convolutional blocks, filters: 32, 64, 128 et 256. ReLu activation. Dropout final à 0.3. Epoch 30. L2 sur Conv2D.

		
		seed 42
		minimum probability : 0.20860421657562256, maximum probability : 0.8079153895378113, average probability : 0.47744637727737427
		AUC: 0.6092705480460582
		threshold : 0.35, accuracy : 0.40476190476190477, precision : 0.39166666666666666, recall : 0.9591836734693877
		threshold : 0.4, accuracy : 0.4708994708994709, precision : 0.411371237458194, recall : 0.8367346938775511
		threshold : 0.45, accuracy : 0.5502645502645502, precision : 0.4433497536945813, recall : 0.6122448979591837
		threshold : 0.5, accuracy : 0.6164021164021164, precision : 0.5076923076923077, recall : 0.4489795918367347
		
		mild sign of overfitting : accuracy = 0.6271 vs val_acc = 0.5909 / auc = 0.6731 vs val_auc = 0.6583

		seed 123

		minimum probability : 0.3004657030105591, maximum probability : 0.39598292112350464, average probability : 0.3374023735523224
		AUC: 0.5892452218982831
		threshold : 0.35, accuracy : 0.6322751322751323, precision : 0.5606060606060606, recall : 0.25170068027210885
		threshold : 0.4, accuracy : 0.6111111111111112, precision : 0, recall : 0.0
		threshold : 0.45, accuracy : 0.6111111111111112, precision : 0, recall : 0.0
		threshold : 0.5, accuracy : 0.6111111111111112, precision : 0, recall : 0.0
		
		Modele collapse. Tout est prédit en bénin

		seed 999	
		minimum probability : 0.3281008005142212, maximum probability : 0.33503684401512146, average probability : 0.3316968083381653
		AUC: 0.5303177548075506
		threshold : 0.35, accuracy : 0.6111111111111112, precision : 0, recall : 0.0
		threshold : 0.4, accuracy : 0.6111111111111112, precision : 0, recall : 0.0
		threshold : 0.45, accuracy : 0.6111111111111112, precision : 0, recall : 0.0
		threshold : 0.5, accuracy : 0.6111111111111112, precision : 0, recall : 0.0
		
		Modele collapse. Tout est prédit en bénin

	5. 4 convolutional blocks, filters: 32, 64, 128 et 256. ReLu activation. Dropout final à 0.3. Epoch 30. AdamW optimizer.
	
		seed 42

		minimum probability : 0.22944091260433197, maximum probability : 0.8156935572624207, average probability : 0.5067823529243469
		AUC: 0.6048531966899313
		threshold : 0.35, accuracy : 0.3915343915343915, precision : 0.38873994638069703, recall : 0.9863945578231292
		threshold : 0.4, accuracy : 0.42063492063492064, precision : 0.39655172413793105, recall : 0.9387755102040817
		threshold : 0.45, accuracy : 0.5079365079365079, precision : 0.42585551330798477, recall : 0.7619047619047619
		threshold : 0.5, accuracy : 0.5740740740740741, precision : 0.4588235294117647, recall : 0.5306122448979592

		mild signs of overfitting : accuracy: 0.6300 vs val_accuracy: 0.6098 / Infirmé par auc: 0.6423 vs val_auc: 0.6329


		seed 123

		minimum probability : 0.2661837637424469, maximum probability : 0.82634437084198, average probability : 0.5160449743270874
		AUC: 0.6215802338251317
		threshold : 0.35, accuracy : 0.3941798941798942, precision : 0.38858695652173914, recall : 0.9727891156462585
		threshold : 0.4, accuracy : 0.43386243386243384, precision : 0.40175953079178883, recall : 0.9319727891156463
		threshold : 0.45, accuracy : 0.47619047619047616, precision : 0.4117647058823529, recall : 0.8095238095238095
		threshold : 0.5, accuracy : 0.5555555555555556, precision : 0.4472361809045226, recall : 0.6054421768707483
	
		accuracy: 0.6271 vs val_accuracy: 0.5985 / auc: 0.6641 vs val_auc: 0.6376


		seed 999

		minimum probability : 0.17550742626190186, maximum probability : 0.6408672332763672, average probability : 0.3417794108390808
		AUC: 0.6315339988809376
		threshold : 0.35, accuracy : 0.6296296296296297, precision : 0.5299145299145299, recall : 0.4217687074829932
		threshold : 0.4, accuracy : 0.6481481481481481, precision : 0.6521739130434783, recall : 0.20408163265306123
		threshold : 0.45, accuracy : 0.6455026455026455, precision : 0.7241379310344828, recall : 0.14285714285714285
		threshold : 0.5, accuracy : 0.626984126984127, precision : 0.6875, recall : 0.07482993197278912

		accuracy: 0.6176 vs val_accuracy: 0.5530 / auc: 0.6544 vs val_auc: 0.6494 

biblio:

• Liao et al. (2024), An open codebase for enhancing transparency in deep learning for breast cancer diagnosis > Baseline, justification
• Wang L. Mammography with deep learning for breast cancer detection. Front Oncol.
2024 Feb 12;14:1281922. doi: 10.3389
https://doi.org/10.3389/fonc.2024.1281922
PMID: 38410114; PMCID: PMC10894909.
• Lee, R., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M. & Rubin, D. L. "A curated
mammography data set for use in computer-aided detection and diagnosis research.",
Sci Data 4, 170177 (2017). https://doi.org/10.1038/sdata.2017.177
• Francois Chollet (2018). Deep Learning with Python. Manning, Shelter Island
• Varoquaux & Cheplygina (2022), Machine learning for medical imaging: methodological failures and recommendations > Data leakage
• Catarino et al. — “The impact of pre-processing techniques on deep learning breast image segmentation”


• Liao et al. (2024), An open codebase for enhancing transparency in deep learning for breast cancer diagnosis
598 x 598 pixel images
crop sur la ROI. Hard mask.
Two sets of models were constructed with different transfer learning techniques :
    - "a couple of convolutional layers, Batch Normalization, and LeakyReLU activation were initially applied to extract features"
    early stopping on the lowest loss
    2 findings:
        ResNet-50 > Xception
        increasing the image size yields better results
"Furthermore, we concerned about the use of denoising and filtering technologies, such as Contrast Limited Adaptive Histogram Equalization, to enhance image quality, as observed in several publications, without adequate validation by trained mammographers. For computational researchers without professional guidance, relying solely on self-implemented mammography image denoising feels akin to navigating blindly and hoping for positive outcomes."


• Varoquaux & Cheplygina (2022), Machine learning for medical imaging: methodological failures and recommendations
Medical datasets are usually small. Larger datasets do not necessarily bring better results
prevent dataset bias > "To assess clinically-relevant predictions, the test data must match the actual target population". In a nutshell, training and test sets should have the same distribution
lots of possible biases
	- Sepctrum bias : the dataset overestimates certain groups
	- Image with interventions (ex pneumothorax) leads to bias
	- Labelling errors
	- Avoid overfitting by observer (distinguish validation and test sets)
Be careful when choosing metrics
	> calibration ?
	> ROC AUC ?
Incorrectly chosen baseline
	- Take a relevant one to measure whether the model improves sth

• Abdikenov, B.; Zhaksylyk, T.; Imasheva, A.; Orazayev, Y.; Karibekov, T. Innovative Multi-View Strategies for AI-Assisted Breast Cancer Detection in Mammography. J. Imaging 2025, 11, 247. https://doi.org/10.3390/jimaging11080247 
	- Merged images CC + MLO > Method = crop black margin until one non-black pixel is spotted. Then, take 5 pixel-margin from there
	- Or, train one CNN on MLO only, one on CC only > features are averaged and passed to additional layers
		"Our findings demonstrate that leveraging both CC and MLO views leads to signifi-
	cantly improved classification performance compared to single-view approaches. In partic-
	ular, the DBE method, which processes each view independently before fusion, achieved
	superior results with architectures such as DenseNet and ResNet50." 
	> Make a case for testing on other db for generalization purposes

https://search.libraries.london.ac.uk/permalink/44SHL_INST/1i2p75j/cdi_doaj_primary_oai_doaj_org_article_27bf4ed8121a415fb1bc5802515273fb

• Shen et al. — “An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization” — Medical Image Analysis, 2021
