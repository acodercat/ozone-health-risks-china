import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker

importance = pd.read_csv("./dataset/importance_GBM.csv")
# 计算比例
importance['proportion'] = importance['importance'] / importance['importance'].sum()
importance = importance.sort_values('proportion', ascending=False)

print(importance)
fig, ax = plt.subplots(figsize=(8, 6))

sns.barplot(x='proportion', y='feature', data=importance, palette='Blues', ax=ax)

plt.subplots_adjust(left=0.25, right=0.75)

ax.set_title('Feature Importance', fontsize=14)
ax.set_xlabel('Importance (%)', fontsize=14)
ax.set_ylabel('Feature', fontsize=14)
fig.savefig('importance_plot.png', bbox_inches='tight', format='png', transparent=True, pad_inches=0, dpi=130)
plt.tight_layout()
plt.show()


#
# importance = importance.sort_values('importance', ascending=False)
#
# fig, ax = plt.subplots(figsize=(8, 6))
#
# sns.barplot(x='importance', y='feature', data=importance, palette='Blues', ax=ax)
#
# plt.subplots_adjust(left=0.25, right=0.75)
#
# ax.set_title('Feature Importance', fontsize=14)
# ax.set_xlabel('Importance', fontsize=14)
# ax.set_ylabel('Feature', fontsize=14)
#
# plt.tight_layout()
# plt.show()

# fig.savefig('importance_plot.png', bbox_inches='tight', format='png', transparent=True, pad_inches=0, dpi=130)
#
# detailed_categories = {
#     "Location": ["Latitude", "Longitude"],
#     "Terrain": ["Elevation", "Ocean"],
#     "Atmospheric Parameters": ["Temp", "Pressure", "Precipitable Water"],
#     "Humidity": ["Spec. Humidity", "Rel. Humidity"],
#     "Wind": ["U-Wind", "Wind Gust"],
#     "AOD": ["AOD"],
#     "Land Use": ["Cultivated Land", "Wood Land", "Grass Land", "Waters", "Urban/Rural", "Unused Land"],
#     "Time Cycle": ["Month", "Year", "Weekday"]
# }
#
# detailed_category_importance = pd.DataFrame(columns=["feature", "importance"])
#
# for category, features in detailed_categories.items():
#     category_importance_sum = importance[importance["feature"].isin(features)]["importance"].sum()
#     detailed_category_importance = pd.concat([detailed_category_importance, pd.DataFrame({"feature": [category], "importance": [category_importance_sum]})], ignore_index=True)
#
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.barplot(x='importance', y='feature', data=detailed_category_importance, palette='Blues', ax=ax)
#
# ax.set_title('Importance of Features by Category', fontsize=18, pad=20)
# ax.set_xlabel('Importance', fontsize=16)
# ax.set_ylabel('Feature', fontsize=16)
#
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
#
# ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#
# sns.despine()
#
# plt.tight_layout()
# plt.show()
#
# fig.savefig('category_importance_plot.png', bbox_inches='tight', format='png', transparent=True, pad_inches=0, dpi=130)
#



