import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Manually creating the DataFrame from the provided data
# In your actual code, you'd read this data in from a CSV or other data file
data = {
    'team': ['Ajax', 'Atalanta', 'Atlético', 'Barcelona', 'Bayern', 'Benfica', 'Beşiktaş', 'Chelsea', 
             'Club Brugge', 'Dortmund', 'Dynamo Kyiv', 'Inter', 'Juventus', 'Leipzig', 'Liverpool', 
             'LOSC', 'Malmö', 'Man. City', 'Man. United', 'Milan', 'Paris', 'Porto', 'Real Madrid', 
             'Salzburg', 'Sevilla', 'Shakhtar Donetsk', 'Sheriff', 'Sporting CP', 'Villarreal', 
             'Wolfsburg', 'Young Boys', 'Zenit'],
    'fouls_committed': [117, 68, 139, 83, 115, 110, 63, 134, 62, 64, 54, 86, 79, 97, 146, 78, 74, 126, 80, 73, 76, 94, 120, 129, 77, 64, 64, 85, 115, 88, 87, 50]
}

df_fouls = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(14, 8))
sns.barplot(x='team', y='fouls_committed', data=df_fouls, palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Fouls Committed by Team')
plt.xlabel('Team')
plt.ylabel('Fouls Committed')
plt.tight_layout()  # Adjusts plot to ensure everything fits without overlapping

# Optionally, display the fouls committed on top of the bars
for index, value in enumerate(df_fouls['fouls_committed']):
    plt.text(index, value, str(value), ha='center', va='bottom')

plt.show()