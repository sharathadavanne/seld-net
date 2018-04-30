# Extracts the features, labels, and normalizes the training and test split features. Make sure you update the location
# of the downloaded datasets before in the cls_feature_class.py

import cls_feature_class

dataset_name = 'echoic'  # echoic, anechoic, circ, circrev, seld, bigseld, bigseldamb
# Extracts feature and labels for all overlap and splits
for ovo in [1, 2, 3]:  # SE overlap
    for splito in [1, 2, 3]:    # all splits
        for nffto in [512]:
            feat_cls = cls_feature_class.FeatureClass(ov=ovo, split=splito, nfft=nffto, echoic=dataset_name)

            # Extract features and normalize them
            feat_cls.extract_all_feature()
            feat_cls.preprocess_features()

            # # Extract labels in regression and classification mode
            feat_cls.extract_all_labels('regr', 0)