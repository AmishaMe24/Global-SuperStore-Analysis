import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.gofplots import qqplot
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import boxcox
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.precision', 2)

sns.set_style("whitegrid")

# ================================
# Data Loading and Preprocessing
# ================================

df = pd.read_csv('Global_Superstore2.csv', encoding='iso-8859-1')

print("\nFirst 5 rows of the dataset are:\n")
print(df.head())

print("\nShape of the dataset is:\n")
print(df.shape)

print("\nChecking for missing values in the dataset:\n")
print(df.isnull().sum())

df.drop("Postal Code",axis=1,inplace=True)

print("\nInformation about the dataset is:\n")
print(df.info())

print("\nDescription of the dataset is:\n")
print(df.describe())

print("\nFirst 5 rows of the dataset after cleaning are:\n")
print(df.head())

print("\nShape of the dataset after cleaning is:\n")
print(df.shape)

print("\nChecking for duplicate values in the dataset:\n")
print(df.duplicated().sum())

print("\nChecking for unique values in the dataset:\n")
print(df.nunique())

df.drop_duplicates(inplace=True)

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

df['Order Year'] = df['Order Date'].dt.year
df['Order Month'] = df['Order Date'].dt.month
df["Ship Year"]=df["Ship Date"].dt.year
df["Ship Month"]=df["Ship Date"].dt.month
df["Unit Price"]=df["Sales"]/df["Quantity"]


df['Order Month'] = df['Order Month'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul',
                                           8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})
df['Ship Month'] = df['Ship Month'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul',
                                         8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})


print("\nFirst 5 rows of the dataset after feature engineering:\n")
print(df.head())

print("\nShape of the dataset after feature engineering:\n")
print(df.shape)

# ================================
# Outlier Detection and Removal
# ================================
numerical_cols = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost', 'Unit Price']

original_df = df.copy()

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower_bound) &
                        (df[col] <= upper_bound)]
    print(f"{col}: Removed outliers beyond [{lower_bound:.2f}, {upper_bound:.2f}]")

fig, axes = plt.subplots(2, 1, figsize=(15, 20))

sns.boxenplot(data=original_df[numerical_cols], ax=axes[0], palette='viridis')
axes[0].set_title('Distribution Before Outlier Removal',
                  fontsize=24, color='blue', fontfamily='serif', fontweight='bold')
axes[0].set_xlabel('Variables', fontsize=20, color='darkred', fontfamily='serif')
axes[0].set_ylabel('Value', fontsize=20, color='darkred', fontfamily='serif')
axes[0].set_xticklabels(labels=numerical_cols, fontsize=18)
axes[0].set_yticklabels(labels=axes[0].get_yticks(), fontsize=18)
axes[0].grid(True, alpha=0.3)

sns.boxenplot(data=df[numerical_cols], ax=axes[1], palette='viridis')
axes[1].set_title('Distribution After Outlier Removal',
                  fontsize=24, color='blue', fontfamily='serif', fontweight='bold')
axes[1].set_xlabel('Variables', fontsize=20, color='darkred', fontfamily='serif')
axes[1].set_ylabel('Value', fontsize=20, color='darkred', fontfamily='serif')
axes[1].set_xticklabels(labels=numerical_cols,fontsize=18)
axes[1].set_yticklabels(labels=axes[1].get_yticks(),fontsize=18)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()

print("\nDataset Summary:")
print(f"Original dataset size: {len(original_df)}")
print(f"Dataset size after outlier removal: {len(df)}")
print(f"Number of outliers removed: {len(original_df) - len(df)}")
print(f"Percentage of data points removed: {((len(original_df) - len(df)) / len(original_df) * 100):.2f}%")

print("\nSummary Statistics Before Outlier Removal:")
print(original_df[numerical_cols].describe())

print("\nSummary Statistics After Outlier Removal:")
print(df[numerical_cols].describe())

print("\nRange Reduction After Outlier Removal:")
for col in numerical_cols:
    original_range = original_df[col].max() - original_df[col].min()
    new_range = df[col].max() - df[col].min()
    reduction_percent = ((original_range - new_range) / original_range) * 100
    print(f"{col}: {reduction_percent:.2f}% reduction in range")


# ================================
# PCA
# ================================
print("\nChecking for missing values in the dataset:\n")
print(df.isnull().sum())

X = df[numerical_cols]
X = (X - X.mean()) / X.std()

pca = PCA(svd_solver="full", random_state=5764, n_components=0.9)
X_pca = pca.fit_transform(X)

print("Original Shape: ", X.shape)
print("Reduced Shape: ", X_pca.shape)

print("Number of features needed to explain more than 90% of the dependent variance:",
    np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0][0] + 1, )

plt.figure(figsize=(10, 10))
plt.plot(
    np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1),
    np.cumsum(pca.explained_variance_ratio_), label="Cumulative Explained Variance",)
plt.xticks(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1))
plt.title("PCA: Cumulative Explained Variance", fontsize=18, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel("Number of Components", fontsize=14, color='darkred', fontfamily='serif')
plt.ylabel("Cumulative Explained Variance", fontsize=14, color='darkred', fontfamily='serif')
plt.grid(True)
plt.axvline(x=(np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0][0] + 1, ),
            color="red", linestyle="--")
plt.axhline(y=0.9, color="black", linestyle="--")
plt.legend()
plt.show()

# Conditional Number of Original and Reduced data
print('Condition Number of Original data:', np.linalg.cond(X))
print('Condition Number of Reduced data:', np.linalg.cond(X_pca))

print('Singular Values:', pca.singular_values_)

# ================================
# Normality Tests
# ================================

# Shapiro-Wilk Test
def shapiro_wilk_test(data, feature, alpha=0.05):
    stat, p = stats.shapiro(data)
    print('=' * 50)
    print(f'Shapiro Test for {feature} feature: Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Sample looks Gaussian/Normal Distribution')
    else:
        print('Sample does not look Gaussian/Normal Distribution')

shapiro_wilk_test(df['Sales'], 'Sales')
shapiro_wilk_test(df['Profit'], 'Profit')
shapiro_wilk_test(df['Quantity'], 'Quantity')
shapiro_wilk_test(df['Discount'], 'Discount')


# D'Agostino's K^2 Test
def dagostino_k2_test(data, feature, alpha=0.05):
    stat, p = stats.normaltest(data)
    print('=' * 50)
    print(f'D\'Agostino\'s K^2 Test for {feature} feature: Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Sample looks Gaussian/Normal Distribution')
    else:
        print('Sample does not look Gaussian/Normal Distribution')

dagostino_k2_test(df['Sales'], 'Sales')
dagostino_k2_test(df['Profit'], 'Profit')
dagostino_k2_test(df['Quantity'], 'Quantity')
dagostino_k2_test(df['Discount'], 'Discount')

# ks-test
def ks_test(data, feature, alpha=0.05):
    stat, p = stats.kstest(data, 'norm')
    print('=' * 50)
    print(f'KS Test for {feature} feature: Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Sample looks Gaussian/Normal Distribution')
    else:
        print('Sample does not look Gaussian/Normal Distribution')

ks_test(df['Sales'], 'Sales')
ks_test(df['Profit'], 'Profit')
ks_test(df['Quantity'], 'Quantity')
ks_test(df['Discount'], 'Discount')

# ==================================
# Data Transformation: Box-Cox
# ==================================

def apply_boxcox(df, features):
    transformed_data = {}
    for feature in features:
        if (df[feature] <= 0).any():
            df[feature] += np.abs(df[feature].min()) + 1

        transformed_data[feature], _ = boxcox(df[feature])
        # make subplot to show orgininal and tranformed data histogram
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        sns.histplot(df[feature], kde=True, ax=axes[0], color='blue', bins=20)
        axes[0].set_title(f'Distribution of {feature}', fontsize=12, color='blue', fontfamily='serif', fontweight='bold')
        axes[0].set_xlabel(feature, fontsize=10, color='darkred', fontfamily='serif')
        axes[0].set_ylabel('Frequency', fontsize=10, color='darkred', fontfamily='serif')

        sns.histplot(transformed_data[feature], kde=True, ax=axes[1], color='blue', bins=20)
        axes[1].set_title(f'Distribution of {feature} after Box-Cox Transformation', fontsize=12, color='blue', fontfamily='serif', fontweight='bold')
        axes[1].set_xlabel(feature, fontsize=10, color='darkred', fontfamily='serif')
        axes[1].set_ylabel('Frequency', fontsize=10, color='darkred', fontfamily='serif')
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(transformed_data, columns=features)

skewed_features = df[numerical_cols].skew()[df[numerical_cols].skew().abs() > 0.5].index
print("\nSkewed Features:\n", skewed_features)
# Apply Box-Cox
transformed_df = apply_boxcox(df, skewed_features)

# Display results
print("Original Data:\n", df.head())
print("\nTransformed Data:\n", transformed_df.head())

# ====================================
# Exploratory Data Analysis (EDA)
# ====================================

def set_plot_labels(title, x_label, y_label):
    plt.title(title, fontsize=15, color='blue', fontfamily='serif', fontweight='bold')
    plt.xlabel(x_label, fontsize=12, color='darkred', fontfamily='serif')
    plt.ylabel(y_label, fontsize=12, color='darkred', fontfamily='serif')
    plt.grid(True)

# ================= Line Plot =======================

sales_by_month = df.groupby('Order Month')['Sales'].sum().reset_index()
profit_by_month = df.groupby('Order Month')['Profit'].sum().reset_index()

plt.figure()
plt.plot(sales_by_month['Order Month'], sales_by_month['Sales'], label='Sales')
set_plot_labels("Sales by Month", "Month", "Sales")
plt.legend()
plt.show()

plt.figure()
plt.plot(profit_by_month['Order Month'], profit_by_month['Profit'], label='Profit')
set_plot_labels("Profit by Month", "Month", "Profit")
plt.legend()
plt.show()

# ======================= Bar Plot =======================

market_sales = df.groupby('Market')['Sales'].sum().sort_values(ascending=False)
plt.figure()
sns.barplot(x=market_sales.index, y=market_sales.values, palette="magma")
set_plot_labels("Market wise Sales", "Market", "Sales")
plt.show()

category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
plt.figure()
sns.barplot(x=category_sales.index, y=category_sales.values, palette="Blues")
set_plot_labels("Category wise Sales", "Category", "Sales")
plt.show()

sub_category_sales = df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(15,14))
sns.barplot(x=sub_category_sales.index, y=sub_category_sales.values, palette="rocket")
plt.title('Sub-Category Wise Sales', fontsize=25, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Sub-Category', fontsize=22, color='darkred', fontfamily='serif')
plt.ylabel('Sales', fontsize=22, color='darkred', fontfamily='serif')
plt.grid(True)
plt.xticks(rotation=90, fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(16, 16))
sns.barplot(x=region_sales.index, y=region_sales.values, palette="cividis")
plt.title('Regions Wise Sales', fontsize=30, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Region', fontsize=22, color='darkred', fontfamily='serif')
plt.ylabel('Sales', fontsize=22, color='darkred', fontfamily='serif')
plt.grid(True)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()

segment_sales = df.groupby('Segment')['Sales'].sum().sort_values(ascending=False)
plt.figure()
sns.barplot(x=segment_sales.index, y=segment_sales.values, palette="viridis")
set_plot_labels("Segment wise Sales", "Segment", "Sales")
plt.show()

category_region_sales = df.groupby(['Category', 'Region'])['Sales'].sum().unstack()
category_region_sales.plot(kind='bar', stacked=True, figsize=(12, 6), color=sns.color_palette("Set2", n_colors=len(category_region_sales.columns)))
plt.title("Stacked Sales by Category and Region", fontsize=16, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel("Category", fontsize=14, color='darkred', fontfamily='serif')
plt.ylabel("Sales", fontsize=14, color='darkred', fontfamily='serif')
plt.legend(title="Region", bbox_to_anchor=(1, 1), loc="upper left", fontsize=12, title_fontsize='13', frameon=True)
plt.grid(True)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

grouped_data = df.groupby(['Category', 'Region'])['Sales'].sum().unstack()

grouped_data.plot(kind='bar', figsize=(10, 6))
plt.title('Sales by Category and Region', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Category', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Sales', fontsize=12, color='darkred', fontfamily='serif')
plt.legend(title="Region", bbox_to_anchor=(1, 1), loc="upper left", fontsize=12, title_fontsize='13', frameon=True)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

stacked_data = df.groupby(['Market', 'Region'])['Sales'].sum().unstack()
stacked_data.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Sales by Market and Region', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Market', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Sales', fontsize=12, color='darkred', fontfamily='serif')
plt.legend(title='Region', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

grouped_data = df.groupby(['Category', 'Sub-Category'])['Profit'].sum().unstack()
grouped_data.plot(kind='bar', figsize=(12, 7))
plt.title('Profit by Category and Sub-Category', fontsize=18, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Category', fontsize=16, color='darkred', fontfamily='serif')
plt.ylabel('Profit', fontsize=16, color='darkred', fontfamily='serif')
plt.legend(title="Sub-Category", bbox_to_anchor=(1, 1), loc="upper left", fontsize=12, title_fontsize='13', frameon=True)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ======================= Count Plot =======================

category_order_count = df['Category'].value_counts()
plt.figure()
sns.barplot(x=category_order_count.index, y=category_order_count.values, palette="viridis")
set_plot_labels("Category wise Order Count", "Category", "Count")
plt.show()

sub_category_order_count = df['Sub-Category'].value_counts()
plt.figure(figsize=(15,14))
sns.barplot(x=sub_category_order_count.index, y=sub_category_order_count.values, palette="plasma")
plt.title('Sub-Category Wise Orders', fontsize=25, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Sub-Category', fontsize=22, color='darkred', fontfamily='serif')
plt.ylabel('Count', fontsize=22, color='darkred', fontfamily='serif')
plt.grid(True)
plt.xticks(rotation=90, fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

region_order_count = df['Region'].value_counts()
plt.figure(figsize=(16, 16))
sns.barplot(x=region_order_count.index, y=region_order_count.values, palette="crest")
plt.title('Regions Wise Orders', fontsize=30, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Region', fontsize=22, color='darkred', fontfamily='serif')
plt.ylabel('Count', fontsize=22, color='darkred', fontfamily='serif')
plt.grid(True)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()

plt.figure()
sns.countplot(data=df, x='Ship Mode', order=df['Ship Mode'].value_counts().index, palette='viridis')
set_plot_labels("Count of Orders by Ship Mode", "Ship Mode", "Count")
plt.show()

plt.figure()
sns.countplot(data=df, x='Ship Mode', hue='Order Priority')
set_plot_labels("Count of Ship Mode by Order Priority", "Ship Mode", "Count")
plt.legend(title='Order Priority')
plt.show()

# ======================= Pie Chart =======================

market_profit = df.groupby('Market')['Profit'].sum()
plt.figure(figsize=(8, 8))
plt.pie(market_profit, labels=market_profit.index, autopct='%1.1f%%', startangle=140,
        colors=plt.cm.Set3(range(len(market_profit))))
plt.title('Profit Contribution by Market', fontsize=16, color='blue', fontfamily='serif', fontweight='bold')
plt.legend()
plt.show()

category_profit = df.groupby('Category')['Profit'].sum()
plt.figure(figsize=(8, 8))
plt.pie(category_profit, labels=category_profit.index, autopct='%1.1f%%', startangle=140,
        colors=plt.cm.Set3(range(len(category_profit))))
plt.title('Total Profit by Category', fontsize=16, color='blue', fontfamily='serif', fontweight='bold')
plt.legend()
plt.show()

country_profit = df.groupby('Country')['Profit'].sum()
top_countries_profit = country_profit.nlargest(10)
plt.figure(figsize=(8, 8))
plt.pie(top_countries_profit, labels=top_countries_profit.index, autopct='%1.1f%%', startangle=140,
        colors=plt.cm.Set3(range(len(top_countries_profit))))
plt.title('Profit by Top 10 Countries', fontsize=16, color='blue', fontfamily='serif', fontweight='bold')
plt.legend(loc='lower left')
plt.show()

# ======================= Dist Plot =======================

numerical_columns = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost', 'Unit Price']

num_cols = len(numerical_columns)
nrows = (num_cols + 2) // 3
fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(18, 5 * nrows))
axes = axes.flatten()

for i, col in enumerate(numerical_columns):
    sns.histplot(df[col], kde=True, ax=axes[i], color='blue', bins=20)
    axes[i].set_title(f'Distribution of {col}', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
    axes[i].set_xlabel(col, fontsize=12, color='darkred', fontfamily='serif')
    axes[i].set_ylabel('Frequency', fontsize=12, color='darkred', fontfamily='serif')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

sns.displot(data=df, x="Discount", hue="Category", kind="kde", fill=True, palette="magma", aspect=1.5, legend=True)
plt.title("Discount Distribution by Category", fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel("Discount", fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel("Density", fontsize=12, color='darkred', fontfamily='serif')
plt.grid(True)
plt.tight_layout()
plt.show()

# ======================= Pair Plot =======================

df_num = df[['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost', 'Unit Price']]

sns.pairplot(df_num, diag_kind='kde')
plt.suptitle('Pair Plot of Global Superstore Dataset', fontsize=16, color='blue', fontfamily='serif', fontweight='bold')
plt.tight_layout()
plt.grid()
plt.show()

# df_num_category = df[['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost', 'Unit Price', 'Category']]
# sns.pairplot(df_num_category, hue='Category', palette='husl')
# plt.suptitle('Pair Plot of Numerical Columns by Category', fontsize=16, color='blue',
#              fontfamily='serif', fontweight='bold')
# plt.tight_layout()
# plt.grid()
# plt.show()

# ======================= Heat Map =======================

correlation_matrix = df_num.corr()
print(f"\nCorrelation Matrix:\n{correlation_matrix}")
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
plt.title('Heatmap of Correlations', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.show()

# ======================= Histogram Plot with KDE =======================

sns.histplot(data=df, x='Sales', kde=True, color='blue', bins=50)
plt.title('Distribution of Sales', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Sales', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Frequency', fontsize=12, color='darkred', fontfamily='serif')
plt.xlim(0, 600)
plt.grid()
plt.tight_layout()
plt.show()

sns.histplot(data=df, x='Profit', kde=True, color='blue', bins=50)
plt.title('Distribution of Profit', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Profit', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Frequency', fontsize=12, color='darkred', fontfamily='serif')
plt.grid()
plt.xlim(0, 150)
plt.tight_layout()
plt.show()

plt.figure()
sns.histplot(data=df, x="Sales", hue="Segment", multiple="stack", kde=True, palette="viridis", edgecolor="black",
             alpha=0.7, bins=50)
plt.title("Stacked Histogram of Sales with KDE by Segment", fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel("Sales", fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel("Frequency", fontsize=12, color='darkred', fontfamily='serif')
plt.grid(axis='y', linestyle="--", alpha=0.7)
plt.xlim(0, 600)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="Market", hue="Region", multiple="stack", legend=True)
plt.title("Stacked Histogram of Market by Region", fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel("Market", fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel("Frequency", fontsize=12, color='darkred', fontfamily='serif')
plt.grid()
plt.tight_layout()
plt.show()

# ======================= QQ- Plot =======================

def plot_qq_before_after_boxcox(data, variable='Sales', figsize=(12, 5)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Original data QQ plot
    qqplot(data[variable], line='s', ax=ax1)
    ax1.set_title(f'QQ Plot Before Box-Cox\nfor {variable}',
                  fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
    ax1.set_xlabel('Theoretical Quantiles', fontsize=12, color='darkred', fontfamily='serif')
    ax1.set_ylabel('Sample Quantiles', fontsize=12, color='darkred', fontfamily='serif')
    ax1.grid(True)

    # Box-Cox transformation
    transformed_data, lambda_param = stats.boxcox(data[variable])

    # QQ plot after Box-Cox transformation
    qqplot(transformed_data, line='s', ax=ax2)
    ax2.set_title(f'QQ Plot After Box-Cox\nfor {variable}\nλ = {lambda_param:.4f}',
                  fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
    ax2.set_xlabel('Theoretical Quantiles', fontsize=12, color='darkred', fontfamily='serif')
    ax2.set_ylabel('Sample Quantiles', fontsize=12, color='darkred', fontfamily='serif')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return lambda_param, transformed_data

lambda_param, transformed_sales = plot_qq_before_after_boxcox(df, 'Sales')

print(f"\nBox-Cox Transformation Summary for Sales:")
print(f"Lambda parameter: {lambda_param:.4f}")
print("\nOriginal Sales Statistics:")
print(f"Skewness: {stats.skew(df['Sales']):.4f}")
print(f"Kurtosis: {stats.kurtosis(df['Sales']):.4f}")
print("\nTransformed Sales Statistics:")
print(f"Skewness: {stats.skew(transformed_sales):.4f}")
print(f"Kurtosis: {stats.kurtosis(transformed_sales):.4f}")

# ======================= KDE Plot with Fill and Customization =======================

sns.kdeplot(df['Profit'], fill=True, alpha=0.6, linewidth=2, palette='viridis')
plt.title('KDE Plot for Profit', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Profit', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Density', fontsize=12, color='darkred', fontfamily='serif')
plt.grid(True)
plt.xlim(0, 150)
plt.tight_layout()
plt.show()

sns.kdeplot(df['Sales'], fill=True, alpha=0.6, linewidth=2, palette='viridis')
plt.title('KDE Plot for Sales', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Sales', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Density', fontsize=12, color='darkred', fontfamily='serif')
plt.grid(True)
plt.xlim(0, 600)
plt.tight_layout()
plt.show()

# ======================= Lm/Reg Plot =======================

sns.regplot(x='Sales', y='Profit', data=df, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.title('Regression Plot', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Sales', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Profit', fontsize=12, color='darkred', fontfamily='serif')
plt.grid()
plt.tight_layout()
plt.show()

sns.lmplot(data=df, x="Sales", y="Profit", hue= 'Category' , fit_reg=False,
palette= 'coolwarm' , height=5, aspect=1)
plt.title("Lm Plot of Sales vs Profit with Category", fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel("Sales", fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel("Profit", fontsize=12, color='darkred', fontfamily='serif')
plt.grid()
plt.tight_layout()
plt.show()

# ======================= Boxen Plot =======================

plt.figure()
sns.boxenplot(data=df[numerical_cols], palette='viridis')
plt.title('Distribution After Outlier Removal',
                  fontsize=16, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Variables', fontsize=12, color='darkred', fontfamily='serif')
plt.xlabel('Value', fontsize=12, color='darkred', fontfamily='serif')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ======================= Area Plot =======================

category_sales = df.pivot_table(index='Order Month',
                              columns='Category',
                              values='Sales',
                              aggfunc='sum')
category_sales.plot.area(stacked=True, alpha=0.7)
plt.title('Monthly Sales by Category - Stacked Area Plot',
          fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Month', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Sales', fontsize=12, color='darkred', fontfamily='serif')
plt.grid(True, alpha=0.3)
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ======================= Violin Plot =======================

sns.violinplot(x='Category', y='Sales', data=df, palette='viridis')
plt.title('Violin Plot: Sales by Category', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Category', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Sales', fontsize=12, color='darkred', fontfamily='serif')
plt.grid()
plt.tight_layout()
plt.show()

sns.violinplot(x='Category', y='Profit', data=df, palette='viridis')
plt.title('Violin Plot: Profit by Category', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Category', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Profit', fontsize=12, color='darkred', fontfamily='serif')
plt.grid()
plt.tight_layout()
plt.show()

# ======================= Joint Plot with KDE and Scatter =======================

sns.jointplot(x='Sales', y='Quantity', data=df, kind='kde')
plt.title('Joint Plot: Sales vs Quantity', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Sales', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Profit', fontsize=12, color='darkred', fontfamily='serif')
plt.grid()
plt.tight_layout()
plt.show()

sns.jointplot(x='Sales', y='Discount', data=df, kind='scatter')
plt.title('Joint Plot: Sales vs Profit', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Sales', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Profit', fontsize=12, color='darkred', fontfamily='serif')
plt.grid()
plt.tight_layout()
plt.show()

# ======================= Rug Plot =======================

sns.scatterplot(x='Sales', y='Profit', data=df, hue='Category', palette='viridis')
sns.rugplot(x='Sales', y='Profit', data=df, hue='Category', palette='viridis')
plt.title('Rug Plot: Sales', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Sales', fontsize=12, color='darkred', fontfamily='serif')
plt.grid(True)
plt.tight_layout()
plt.show()

# ======================= 3D Plot and Contour Plot =======================

sales = np.linspace(df['Sales'].min(), df['Sales'].max(), 50)
profit = np.linspace(df['Profit'].min(), df['Profit'].max(), 50)
X, Y = np.meshgrid(sales, profit)

from scipy.interpolate import griddata
Z = griddata((df['Sales'], df['Profit']), df['Quantity'], (X, Y), method='linear')
fig = plt.figure(figsize=(12, 8))  # Increase figure size
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none', alpha=0.6)

contours = ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap='coolwarm',
                     linewidths=2, levels=15)

ax.contour(X, Y, Z, zdir='x', offset=X.min(), cmap='coolwarm', linewidths=2)
ax.contour(X, Y, Z, zdir='y', offset=Y.max(), cmap='coolwarm', linewidths=2)

ax.set_xlabel("Sales", fontsize=12, family="serif", color="darkred")
ax.set_ylabel("Profit", fontsize=12, family="serif", color="darkred")
ax.set_zlabel("Quantity", fontsize=12, family="serif", color="darkred")
ax.set_title("3D Surface and Contour Plot for Sales, Profit, Quantity",
             fontsize=16, family="serif", color="blue", fontweight='bold')

plt.colorbar(surf, shrink=0.5, aspect=10)

ax.view_init(elev=30, azim=45)

plt.show()

# ======================= Cluster Plot =======================

sns.clustermap(df_num.sample(n=1000, random_state=5764),
               cmap='vlag',
               method='single',
               standard_scale=1)
plt.title('Cluster Map', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.tight_layout()
plt.show()


# ======================= Hexbin Plot =======================

df.plot.hexbin(x='Sales', y='Profit', gridsize=20, cmap='viridis')
plt.title('Hexbin Plot of Sales and Profit', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Sales', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Profit', fontsize=12, color='darkred', fontfamily='serif')
plt.grid()
plt.tight_layout()
plt.show()

# ======================= Strip Plot =======================

plt.figure()
sns.stripplot(x='Category', y='Sales', data=df, palette='viridis')
plt.title('Strip Plot: Sales by Category', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Category', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Sales', fontsize=12, color='darkred', fontfamily='serif')
plt.grid()
plt.tight_layout()
plt.show()

plt.figure()
sns.stripplot(data=df, x='Category', y='Sales', hue='Segment',
             palette='Set2', jitter=0.2, size=5, alpha=0.6)
plt.title('Sales by Category (Colored by Segment)',
         fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Category', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Sales', fontsize=12, color='darkred', fontfamily='serif')
plt.legend(title='Segment', bbox_to_anchor=(1.05, 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ======================= Swarm Plot =======================

sampled_df = df.sample(n=1000, random_state=5764)

plt.figure()
sns.swarmplot(x='Category', y='Sales', data=sampled_df, palette='viridis')
plt.title('Swarm Plot: Sales by Category', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Category', fontsize=12, color='darkred', fontfamily='serif')
plt.ylabel('Profit', fontsize=12, color='darkred', fontfamily='serif')
plt.grid()
plt.tight_layout()
plt.show()

# ======================= Sub Plot =======================

# subplot
plt.figure(figsize=(20, 15))
plt.suptitle('Business Performance Story Through Pie Charts',
             fontsize=24, color='blue', fontfamily='serif', fontweight='bold', y=0.95)

# 1. Revenue Distribution
plt.subplot(2, 2, 1)
category_sales = df.groupby('Category')['Sales'].sum()
plt.pie(category_sales, labels=category_sales.index, autopct='%1.1f%%',
        colors=sns.color_palette('Set3'),
        wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
        textprops={'fontsize': 14, 'color': 'darkblue'})
plt.title('Revenue Distribution by Category\nTotal Revenue: ${:,.0f}'.format(category_sales.sum()),
          pad=20, fontsize=18, color='blue', fontfamily='serif', fontweight='bold')

# 2. Profit Contribution with Inner Circle
plt.subplot(2, 2, 2)
segment_profit = df.groupby('Segment')['Profit'].sum()
category_profit = df.groupby('Category')['Profit'].sum()

# Create donut chart
plt.pie(segment_profit, labels=segment_profit.index, autopct='%1.1f%%',
        colors=sns.color_palette('Pastel1'),
        wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
        textprops={'fontsize': 14, 'color': 'darkblue'}, radius=1)
plt.pie([1], colors=['white'], radius=0.7)
plt.title('Profit Distribution by Segment\nTotal Profit: ${:,.0f}'.format(segment_profit.sum()),
          pad=20, fontsize=18, color='blue', fontfamily='serif', fontweight='bold')

# 3. Regional Market Share with Exploded Segments
plt.subplot(2, 2, 3)
region_orders = df.groupby('Region').size()
explode = [0.05] * len(region_orders)

plt.pie(region_orders, labels=region_orders.index, autopct='%1.1f%%',
        explode=explode, colors=sns.color_palette('Spectral'),
        wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
        textprops={'fontsize': 14, 'color': 'darkblue'})
plt.title('Order Distribution by Region\nTotal Orders: {:,}'.format(len(df)),
          pad=20, fontsize=18, color='blue', fontfamily='serif', fontweight='bold')

# 4. Shipping Analysis - Nested Pie Chart
plt.subplot(2, 2, 4)
ship_mode = df.groupby('Ship Mode')['Shipping Cost'].sum()
priority = df.groupby('Order Priority')['Shipping Cost'].sum()

# Outer pie chart for shipping mode
plt.pie(ship_mode, labels=ship_mode.index, autopct='%1.1f%%',
        colors=sns.color_palette('Set2'),
        wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
        textprops={'fontsize': 14, 'color': 'darkblue'}, radius=1)

# Inner pie chart for order priority
plt.pie(priority, labels=priority.index, autopct='%1.1f%%',
        colors=sns.color_palette('Pastel2'),
        wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
        textprops={'fontsize': 12, 'color': 'darkblue'}, radius=0.7)

plt.title('Shipping Cost Distribution\nOuter: Ship Mode, Inner: Order Priority\nTotal Shipping Cost: ${:,.0f}'.format(ship_mode.sum()),
          pad=20, fontsize=18, color='blue', fontfamily='serif', fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# subplot
grouped_sales = df.groupby(['Category', 'Order Year'])['Sales'].sum().unstack()
grouped_profit = df.groupby(['Category', 'Order Year'])['Profit'].sum().unstack()

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
grouped_sales.plot(kind='bar', ax=ax[0])
grouped_profit.plot(kind='bar', ax=ax[1])

ax[0].set_title('Sales by Category and Year', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')
ax[1].set_title('Profit by Category and Year', fontsize=14, color='blue', fontfamily='serif', fontweight='bold')

for a in ax:
    a.set_xlabel('Category', fontsize=12, color='darkred', fontfamily='serif')
    a.set_ylabel('Value', fontsize=12, color='darkred', fontfamily='serif')
    a.legend(title='Year', fontsize=10)
    a.grid(axis='y', linestyle='--', alpha=0.7)
    a.set_xticks(range(len(grouped_sales.index)))
    a.set_xticklabels(grouped_sales.index, rotation=0)

plt.tight_layout()
plt.show()

# subplot
# 1. Profitability and Performance Analysis
plt.figure(figsize=(20, 12))
plt.suptitle('Profitability and Performance Insights',
             fontsize=24, color='blue', fontfamily='serif', fontweight='bold')

# Profit Distribution by Category and Segment
plt.subplot(2, 2, 1)
avg_profit = df.groupby(['Category', 'Segment'])['Profit'].mean().unstack()
sns.heatmap(avg_profit, annot=True, fmt='.0f', cmap='RdYlGn', center=0)
plt.title('Average Profit per Order ($)', fontsize=18, pad=10, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Customer Segment', fontsize=16, color='darkred', fontfamily='serif')
plt.ylabel('Category', fontsize=16, color='darkred', fontfamily='serif')

plt.subplot(2, 2, 2)
df['Discount_Band'] = pd.cut(df['Discount'],
                            bins=[-np.inf, 0, 0.2, 0.4, np.inf],
                            labels=['No Discount', 'Low', 'Medium', 'High'])

discount_impact = df.groupby('Discount_Band').agg({
    'Profit': 'mean',
    'Sales': 'mean'
}).reset_index()

ax1 = plt.gca()
ax2 = ax1.twinx()
sns.barplot(data=discount_impact, x='Discount_Band', y='Sales', color='skyblue', alpha=0.5, ax=ax1)
sns.lineplot(data=discount_impact, x='Discount_Band', y='Profit', color='red', marker='o', linewidth=2, ax=ax2)
ax1.set_title('Impact of Discounts on Sales and Profit', fontsize=18, pad=10, color='blue', fontfamily='serif', fontweight='bold')
ax1.set_xlabel('Discount Level', fontsize=14)
ax1.set_ylabel('Average Sales ($)', fontsize=14, color='skyblue')
ax2.set_ylabel('Average Profit ($)', fontsize=14, color='red')
ax1.grid(True, alpha=0.3)

# Regional Performance
plt.subplot(2, 2, 3)
regional_metrics = df.groupby('Region').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count'
}).reset_index()
regional_metrics['Profit_Margin'] = (regional_metrics['Profit'] / regional_metrics['Sales']) * 100

plt.scatter(regional_metrics['Sales'], regional_metrics['Profit'],
           s=regional_metrics['Order ID']/10, alpha=0.6, c=regional_metrics['Profit_Margin'],
           cmap='viridis')
for i, region in enumerate(regional_metrics['Region']):
    plt.annotate(region,
                (regional_metrics['Sales'].iloc[i], regional_metrics['Profit'].iloc[i]),
                xytext=(5, 5), textcoords='offset points')
plt.colorbar(label='Profit Margin (%)')
plt.title('Regional Performance Overview', fontsize=18, pad=10, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Total Sales ($)', fontsize=14, color='darkred', fontfamily='serif')
plt.ylabel('Total Profit ($)', fontsize=14, color='darkred', fontfamily='serif')
plt.grid(True, alpha=0.3)

# Shipping Analysis
plt.subplot(2, 2, 4)
ship_metrics = df.groupby(['Ship Mode', 'Order Priority']).agg({
    'Shipping Cost': 'mean',
    'Order ID': 'count'
}).reset_index()

sns.scatterplot(data=ship_metrics, x='Shipping Cost', y='Order ID',
                hue='Ship Mode', size='Order ID', sizes=(100, 1000),
                style='Order Priority', palette='deep')
plt.title('Shipping Cost vs Order Volume', fontsize=18, pad=10, color='blue', fontfamily='serif', fontweight='bold')
plt.xlabel('Average Shipping Cost ($)', fontsize=14, color='darkred', fontfamily='serif')
plt.ylabel('Number of Orders', fontsize=14, color='darkred', fontfamily='serif')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# subplot
fig = plt.figure(figsize=(20, 15))
plt.suptitle('Global Superstore Performance Story',
             fontsize=24, color='blue', fontfamily='serif', fontweight='bold')

# 1. Sales and Profit by Category and Segment
ax1 = plt.subplot(2, 2, 1)
segment_category = pd.crosstab(df['Category'], df['Segment'],
                              values=df['Sales'], aggfunc='sum', normalize='index')
sns.heatmap(segment_category, annot=True, fmt='.1%', cmap='YlOrRd', ax=ax1)
ax1.set_title('Category Sales Distribution by Segment',
              fontsize=18, color='blue', fontfamily='serif', fontweight='bold')
ax1.set_xlabel('Customer Segment', fontsize=14, color='darkred', fontfamily='serif')
ax1.set_ylabel('Product Category', fontsize=14, color='darkred', fontfamily='serif')

# 2. Regional Performance with Ship Mode
ax2 = plt.subplot(2, 2, 2)
region_ship = pd.crosstab(df['Region'], df['Ship Mode'],
                         values=df['Sales'], aggfunc='sum')
region_ship_pct = region_ship.div(region_ship.sum(axis=1), axis=0)
region_ship_pct.plot(kind='bar', stacked=True, ax=ax2,
                     colormap='Set3', width=0.8)
ax2.set_title('Regional Sales by Shipping Mode',
              fontsize=18, color='blue', fontweight='bold')
ax2.set_xlabel('Region', fontsize=14, color='darkred', fontfamily='serif')
ax2.set_ylabel('Percentage of Sales', fontsize=14, color='darkred', fontfamily='serif')
ax2.legend(title='Ship Mode')
ax2.grid(True, alpha=0.3)

# 3. Order Priority and Shipping Cost Analysis
ax3 = plt.subplot(2, 2, 3)
priority_metrics = df.groupby('Order Priority').agg({
    'Shipping Cost': 'mean',
    'Sales': 'mean'
}).reset_index()

bar_width = 0.35
x = np.arange(len(priority_metrics['Order Priority']))

ax3_twin = ax3.twinx()
bars1 = ax3.bar(x - bar_width/2, priority_metrics['Shipping Cost'],
                bar_width, label='Avg Shipping Cost', color='skyblue')
bars2 = ax3_twin.bar(x + bar_width/2, priority_metrics['Sales'],
                     bar_width, label='Avg Sales', color='lightgreen')

ax3.set_xticks(x)
ax3.set_xticklabels(priority_metrics['Order Priority'])
ax3.set_ylabel('Average Shipping Cost ($)', fontsize=14, color='darkred', fontfamily='serif')
ax3_twin.set_ylabel('Average Sales ($)', fontsize=14, color='darkred', fontfamily='serif')

# Add both legends
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax3.set_title('Order Priority: Shipping Cost vs Sales',
              fontsize=18, color='blue', fontweight='bold', fontfamily='serif')
ax3.grid(True, alpha=0.3)

# 4. Category Performance Over Time
ax4 = plt.subplot(2, 2, 4)
monthly_category = df.pivot_table(index='Order Month',
                                columns='Category',
                                values='Sales',
                                aggfunc='sum')
monthly_category.plot(marker='o', ax=ax4, linewidth=2)
ax4.set_title('Monthly Sales Trend by Category',
              fontsize=18, color='blue', fontweight='bold', fontfamily='serif')
ax4.set_xlabel('Month', fontsize=14, color='darkred', fontfamily='serif')
ax4.set_ylabel('Total Sales ($)', fontsize=14, color='darkred', fontfamily='serif')
ax4.legend(title='Category')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =======================
# Tables
# =======================

from prettytable import PrettyTable

# 1. Category Performance Summary
category_summary = df.groupby('Category').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count',
    'Discount': 'mean'
}).round(2)

category_summary['Profit_Margin'] = (category_summary['Profit'] / category_summary['Sales'] * 100).round(2)
category_summary['Avg_Order_Value'] = (category_summary['Sales'] / category_summary['Order ID']).round(2)

cat_table = PrettyTable()
cat_table.title = "Category Performance Overview"
cat_table.field_names = ["Category", "Total Sales ($)", "Total Profit ($)", "Orders",
                        "Profit Margin (%)", "Avg Order Value ($)", "Avg Discount (%)"]

for category in category_summary.index:
    cat_table.add_row([
        category,
        f"{category_summary.loc[category, 'Sales']:,.2f}",
        f"{category_summary.loc[category, 'Profit']:,.2f}",
        f"{category_summary.loc[category, 'Order ID']:,}",
        f"{category_summary.loc[category, 'Profit_Margin']:,.2f}",
        f"{category_summary.loc[category, 'Avg_Order_Value']:,.2f}",
        f"{category_summary.loc[category, 'Discount']*100:,.2f}"
    ])

# 2. Regional Shipping Analysis
shipping_summary = df.groupby(['Region', 'Ship Mode']).agg({
    'Order ID': 'count',
    'Shipping Cost': ['mean', 'sum'],
    'Sales': 'sum'
}).round(2)

shipping_summary.columns = ['Orders', 'Avg_Shipping_Cost', 'Total_Shipping_Cost', 'Total_Sales']
shipping_summary['Shipping_to_Sales_Ratio'] = (shipping_summary['Total_Shipping_Cost'] /
                                             shipping_summary['Total_Sales'] * 100).round(2)
shipping_summary = shipping_summary.reset_index()

ship_table = PrettyTable()
ship_table.title = "Regional Shipping Analysis"
ship_table.field_names = ["Region", "Ship Mode", "Orders", "Avg Shipping Cost ($)",
                         "Total Shipping Cost ($)", "Shipping/Sales Ratio (%)"]

for _, row in shipping_summary.iterrows():
    ship_table.add_row([
        row['Region'],
        row['Ship Mode'],
        f"{row['Orders']:,}",
        f"{row['Avg_Shipping_Cost']:,.2f}",
        f"{row['Total_Shipping_Cost']:,.2f}",
        f"{row['Shipping_to_Sales_Ratio']:,.2f}"
    ])

segment_summary = df.groupby(['Segment', 'Category']).agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count',
    'Discount': 'mean'
}).round(2)

segment_summary['Profit_per_Order'] = (segment_summary['Profit'] /
                                     segment_summary['Order ID']).round(2)
segment_summary = segment_summary.reset_index()

# Create PrettyTable for Segment Summary
seg_table = PrettyTable()
seg_table.title = "Customer Segment Profitability by Category"
seg_table.field_names = ["Segment", "Category", "Total Sales ($)", "Total Profit ($)",
                        "Orders", "Profit per Order ($)", "Avg Discount (%)"]

for _, row in segment_summary.iterrows():
    seg_table.add_row([
        row['Segment'],
        row['Category'],
        f"{row['Sales']:,.2f}",
        f"{row['Profit']:,.2f}",
        f"{row['Order ID']:,}",
        f"{row['Profit_per_Order']:,.2f}",
        f"{row['Discount']*100:,.2f}"
    ])

for table in [cat_table, ship_table, seg_table]:
    table.align = 'r'
    table.align['Category'] = table.align['Region'] = table.align['Segment'] = table.align['Ship Mode'] = 'l'
    table.float_format = '.2'

print("\nTable 1: Category Performance")
print(cat_table)
print("\nTable 2: Regional Shipping Analysis")
print(ship_table)
print("\nTable 3: Customer Segment Profitability")
print(seg_table)

