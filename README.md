# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("/content/42_Cases_under_crime_against_women.csv",
                 encoding='latin1',
                 on_bad_lines='skip')

# ===============================
# CLEAN DATA
# ===============================
df.dropna(inplace=True)
df.columns = df.columns.str.strip()

# ===============================
# DETECT COLUMNS
# ===============================
state_col = None
year_col = None

for col in df.columns:
    if 'state' in col.lower():
        state_col = col
    if 'year' in col.lower():
        year_col = col

# ===============================
# CREATE TOTAL COLUMN
# ===============================
if 'Total' not in df.columns:
    numeric_cols = df.select_dtypes(include='number').columns
    df['Total'] = df[numeric_cols].sum(axis=1)

# ===============================
# 1.  STATES (BAR)
# ===============================
if state_col:
    top_states = df.groupby(state_col)['Total'].sum().nlargest(5)

    plt.figure()
    top_states.plot(kind='bar')
    plt.title("Top 5 States")
    plt.show()

# ===============================
# 2.  YEARS (LINE)
# ===============================
if year_col:
    top_years = df.groupby(year_col)['Total'].sum().nlargest(5)

    plt.figure()
    top_years.sort_index().plot(marker='o')
    plt.title("Top 5 Years")
    plt.show()

# ===============================
# 3.CRIME TYPES (BAR)
# ===============================
crime_types = df.select_dtypes(include='number').sum().nlargest(5)

plt.figure()
crime_types.plot(kind='bar')
plt.title("Top 5 Crime Types")
plt.show()

# ===============================
# 4. HEATMAP 
# ===============================
if state_col and year_col:
    pivot = df.pivot_table(values='Total', index=state_col, columns=year_col)

    top_states = pivot.sum(axis=1).nlargest(5).index
    top_years = pivot.sum(axis=0).nlargest(5).index

    pivot = pivot.loc[top_states, top_years]

    plt.figure()
    sns.heatmap(pivot)
    plt.title("Top 5 Heatmap")
    plt.show()

# ===============================
# 5. CORRELATION 
# ===============================
corr = df.corr(numeric_only=True)

cols = corr.sum().nlargest(5).index
corr_top5 = corr.loc[cols, cols]

plt.figure()
sns.heatmap(corr_top5, annot=True)
plt.title("Top 5 Correlation")
plt.show()

# ===============================
# 6. BOXPLOT 
# ===============================
top5_total = df['Total'].nlargest(5)

plt.figure()
sns.boxplot(x=top5_total)
plt.title("Boxplot (Top 5)")
plt.show()

# ===============================
# 7. PIE CHART 
# ===============================
if state_col:
    plt.figure()
    top_states.plot(kind='pie', autopct='%1.1f%%')
    plt.title("Top 5 State Share")
    plt.ylabel("")
    plt.show()

# ===============================
# 8. HISTOGRAM 
# ===============================
plt.figure()
top5_total.plot(kind='hist')
plt.title("Histogram (Top 5)")
plt.show()

# ===============================
# 9. SCATTER 
# ===============================
if year_col:
    sample = df.nlargest(5, 'Total')

    plt.figure()
    plt.scatter(sample[year_col], sample['Total'])
    plt.title("Top 5 Scatter")
    plt.show()

# ===============================
# 10. TREND 
# ===============================
if year_col:
    yearly = df.groupby(year_col)['Total'].sum().nlargest(5)

    plt.figure()
    plt.plot(yearly.sort_index(), marker='o')
    plt.title("Top 5 Trend")
    plt.show()
