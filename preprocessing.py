from autoarm.preprocessing import FeatureSelection

fs = FeatureSelection("datasets/dataset1.csv")

fs.load_dataset()

#print (fs.features)

selected_features = fs.feature_selection_threshold()

print (selected_features)
