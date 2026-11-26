import numpy as np
import pandas as pd
def loaddata(filepath):
    data=pd.read_csv(filepath)
    return data
if __name__=="__main__":    
    data=loaddata("data/raw/Crop_recommendation.csv")
    print(data.head())
    print(data.shape)
features=['N','P','K','temperature','humidity','ph','rainfall']
def calculatecentroids(data):
    x=data[features].to_numpy(dtype=float)
    labels=data['label'].to_numpy()
    names_crop=np.unique(labels)
    membership=labels[:,None]==names_crop[None,:]
    counts=membership.sum(axis=0)
    centroids=(membership.T@x)/counts[:,None]
    feature_mean=x.mean(axis=0)
    feature_std=x.std(axis=0)
    if np.any(feature_std==0):
        raise ValueError("A Feature  has zero stand deviation")

    return names_crop, centroids, feature_mean, feature_std
def standardize(centroids,feature_mean,feature_std):
    std=(centroids-feature_mean)/feature_std
    return std
def verify(data,centroids):
    reference=data.groupby('label',sort=True)[features].mean().to_numpy()
    return np.allclose(reference,centroids)
###verification
if __name__=="__main__":    
    names_crop, centroids, feature_mean, feature_std=calculatecentroids(data)
    std=standardize(centroids,feature_mean,feature_std)
    print("VECTORIZED FEATURE DERIVATION\n")
    print("Shape of dataset:",data.shape)
    print("Crop_number:",len(names_crop))
    print("Matrix shape:",data[features].to_numpy().shape)
    print("Centroids shape:",centroids.shape)
    print("Standardized centroids shape:",std.shape)
    print("Feature matrix size:",data[features].to_numpy().nbytes,"bytes")
    print("Centroids size:",centroids.nbytes)
    print("Standardized centroids size:",std.nbytes)
    print("Verification of centroids:",verify(data,centroids))
    print("Standardized centroids:\n")
    print(np.round(std,3))
##Q5

def identify_water_crops(data):
    names_crop=np.unique(data['label'].to_numpy())
    labels=data['label'].to_numpy()
    rainfall=data['rainfall'].to_numpy(dtype=float)
    membership = labels[:, None] == names_crop[None, :]
    counts = membership.sum(axis=0)
    average_rainfall = (membership.T @ rainfall) / counts
    high_rainfall_mask = average_rainfall > 200
    high_rainfall_crops = names_crop[high_rainfall_mask]
    return names_crop, average_rainfall, high_rainfall_mask, high_rainfall_crops
def verify_pulse_separation(crop_names, high_rainfall_mask):
    pulse_crops = {
        'chickpea', 'kidneybeans', 'pigeonpeas',
        'mothbeans', 'mungbean', 'blackgram', 'lentil'
    }

    pulse_mask = np.isin(crop_names, list(pulse_crops))

    overlap_mask = high_rainfall_mask & pulse_mask

    return pulse_mask, overlap_mask


names_crop, average_rainfall, high_rainfall_mask, high_rainfall_crops = identify_water_crops(data)

pulse_mask, overlap_mask = verify_pulse_separation(names_crop, high_rainfall_mask)

print("\nHIGH-WATER VS DROUGHT-RESISTANT CROPS")
print("\nHigh-rainfall crops (average rainfall > 200 mm):")
print(high_rainfall_crops)

print("\nDrought-tolerant pulse crops:")
print(names_crop[pulse_mask])

print("\nAverage rainfall of each crop:")
for i in range(len(names_crop)):
    print(names_crop[i], ":", round(average_rainfall[i], 2), "mm")

print("\nVerification of separation:", not np.any(overlap_mask))
print("Overlapping crops:",names_crop[overlap_mask])


def prepare_ols(data):
    cereal_crops=['rice','maize']
    cereal_data=data[data['label'].isin(cereal_crops)]
    features_predictors=['temperature','humidity']
    x=cereal_data[features_predictors].to_numpy(dtype=float)
    y=cereal_data['N'].to_numpy(dtype=float)
    x=np.column_stack((np.ones(len(x)),x))
    return x,y,features_predictors
def calculateols(x,y):
    beta=np.linalg.lstsq(x,y,rcond=None)[0]
    return beta
def evaluateols(x,y,beta):
    y_pred=x@beta
    residuals=y-y_pred
    rss=np.sum(residuals**2)
    tss=np.sum((y-y.mean())**2)
    r_sq=1-(rss/tss)
    return y_pred,residuals,rss,tss,r_sq
x,y,features_predictors=prepare_ols(data)
beta=calculateols(x,y)
y_pred,residuals,rss,tss,r_sq=evaluateols(x,y,beta)
print("first principle ols")
print("\n")
print("Design matrix shape:",x.shape)
print("Target matrix shape:",y.shape)
print("Regression coefficiants:")
print("Intercept:",beta[0])
for i in range(len(features_predictors)):
    print(features_predictors[i],":",beta[i+1])
print("Predicted NITROGEN_of_5records:")
print(y_pred[:5])
print("Residuals for 5 records:",residuals[:5])
print("RSS:",rss)
print("TSS:",tss)
print("R-squared:",r_sq)

##q7
### Q7

# Count samples for each crop
sample_counts=data.groupby('label').size()

print("\nSAMPLE COUNT PER CROP")
print(sample_counts)
print("\nTotal samples:",sample_counts.sum())
print("All crops have 100 samples:",(sample_counts==100).all())

# Define plantation and pulse crop groups
plantation_crops={'coconut','coffee'}
pulse_crops={
    'chickpea','kidneybeans','pigeonpeas',
    'mothbeans','mungbean','blackgram','lentil'
}

# Select only plantation and pulse crops
grouped_data=data[
    data['label'].isin(plantation_crops | pulse_crops)
].copy()

# Assign a group name to each crop
grouped_data['crop_group']=np.where(
    grouped_data['label'].isin(plantation_crops),
    'Plantation',
    'Pulse'
)

# Calculate IQR
def calculate_iqr(s):
    return s.quantile(0.75)-s.quantile(0.25)

# Calculate variance and IQR for N, P and K
feature_audit=grouped_data.groupby('crop_group')[['N','P','K']].agg(
    ['var',calculate_iqr]
)

print("\nVARIANCE AND IQR OF N, P AND K")
print(feature_audit)
####Q8

####Q8

def discretize(data):
    categorized_data = data.copy()

    # Categorize soil pH
    categorized_data['ph_category'] = np.select(
        [
            categorized_data['ph'] < 6.0,
            categorized_data['ph'] <= 7.5
        ],
        ['Acidic', 'Neutral'],
        default='Alkaline'
    )

    # Categorize rainfall into three equal-width ranges
    categorized_data['rainfall_category'] = pd.cut(
        categorized_data['rainfall'],
        bins=3,
        labels=['Low', 'Medium', 'High']
    )

    return categorized_data

categorized_data = discretize(data)

print("\nDISCRETIZED SOIL pH AND RAINFALL")
print(categorized_data[
    ['ph', 'ph_category', 'rainfall', 'rainfall_category']
].head(10))

# Pivot table showing the dominant crop
pivot_table = pd.pivot_table(
    categorized_data,
    index='ph_category',
    columns='rainfall_category',
    values='label',
    aggfunc=lambda x: x.mode().iloc[0],
    margins=True,
    observed=False
)

print("\nPIVOT TABLE - DOMINANT CROPS")
print(pivot_table)

print("\nDISTRIBUTION SUMMARY")
print("Total records:", len(categorized_data))
print("\npH category counts:")
print(categorized_data['ph_category'].value_counts())

print("\nRainfall category counts:")
print(categorized_data['rainfall_category'].value_counts())

print("\nTotal crops:", categorized_data['label'].nunique())


# Q9

def reshapetolong(data):
    long_data = data.copy()
    long_data['record_id'] = np.arange(len(long_data))

    long_data = pd.melt(
        long_data,
        id_vars=['record_id', 'label'],
        value_vars=['N', 'P', 'K'],
        var_name='attribute',
        value_name='value'
    )

    return long_data

long_data = reshapetolong(data)

print("\nRESHAPED NUTRIENT DATA")
print(long_data.head(10))
print("Long format shape:", long_data.shape)
print("Columns:", long_data.columns.tolist())


def rank_top_crops(long_data, top_n=3):
    average_values = (
        long_data.groupby(['attribute', 'label'])['value']
        .mean()
        .reset_index(name='average_requirement')
    )

    ranked_data = (
        average_values.sort_values(
            ['attribute', 'average_requirement'],
            ascending=[True, False]
        )
        .groupby('attribute')
        .head(top_n)
    )

    return ranked_data

ranked_summary = rank_top_crops(long_data, top_n=3)

print("\nTOP THREE CROPS FOR EACH NUTRIENT")
print(
    ranked_summary[
        ['attribute', 'label', 'average_requirement']
    ].to_string(index=False)
)

print("\nTotal ranked results:", len(ranked_summary))
