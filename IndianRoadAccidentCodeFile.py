#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[86]:


df = pd.read_csv(r"C:\Users\Manish bhikonde\Downloads\Indian_Road_accident\indian_roads_dataset.csv")


# In[88]:


df


# In[90]:


df.info()


# In[92]:


df.isnull().sum()


# In[94]:


df.count()


# In[96]:


df.describe()


# In[98]:


df.fillna({
    'weather': 'unknown',
    'visibility': 'unknown',
    'traffic_density': 'unknown'
}, inplace=True)


# In[74]:


df['accident_severity'] = df['accident_severity'].map({
    'minor': 0,
    'major': 1,
    'fatal': 2
})


# In[100]:


pd.pivot_table(df, values='accident_severity', index='city', aggfunc='count')


# In[102]:


df.groupby('city').size().reset_index(name='Total_Accidents')


# In[104]:


severity_count = df['accident_severity'].value_counts().reset_index()
severity_count.columns = ['Severity', 'Count']
severity_count


# In[187]:


df.groupby(['risk_score', 'vehicles_involved']).size().reset_index(name='Count')


# In[106]:


df.groupby(['city', 'accident_severity']).size().reset_index(name='Count')


# #### Accidents by city

# In[109]:


plt.figure()
df['city'].value_counts().plot(kind='bar')
plt.title("Accidents by City")
plt.xlabel("City")
plt.ylabel("Count")
plt.xticks(rotation=65)
plt.show()


# Most accident city is - Chandigarh, Chennai, and Kolkata

# #### Accident Severity

# In[130]:


df['accident_severity'].value_counts().plot(kind='pie', autopct='%.1f%%')
plt.show()


# This pie chart represents the percentage distribution of accident severity levels.
# Minor Accident:- 55.1%
# Major Accident:- 29.9%
# Spot death:- 14.9% 

# #### Accident by weather

# In[133]:


plt.figure()
sns.countplot(x='weather', data=df)
plt.title("Accidents by Weather Condition")
plt.xticks(rotation=45)
plt.show()


# This chart shows the count of each weather category

# #### Accident by Road type

# In[136]:


plt.figure()
sns.countplot(x='road_type', data=df)
plt.title("Accidents by Road Type")
plt.show()


# Accident hua tb road kaisi thi 

# Traffic density

# In[139]:


plt.figure()
sns.countplot(x='traffic_density', data=df)
plt.title("Traffic Density vs Accidents")
plt.show()


# Accident hua tb traffic kaise tha.
# Most of accident in high traffic and low traffic.
# Medium traffic me sabse kam accident huye hai.

# Day of week 

# In[148]:


plt.figure()
sns.countplot(x='day_of_week', data=df)
plt.title("Accidents by Day of Week")
plt.xticks(rotation=45)
plt.show()


# Week ke kis din Sabse accident huye hai.
# Sabse jyada accident `Monday` ke din huye hai 

# Vehical involve

# In[152]:


plt.figure()
sns.boxplot(x=df['vehicles_involved'])
plt.title("Vehicles Involved Distribution")
plt.show()


# Most of accident me 3 vehical involved hai

# Risk Score

# In[175]:


plt.figure()
sns.histplot(df['risk_score'], kde=True)
plt.title("Risk Score Distribution")
plt.show()


# The chart shows that accidents occurred even in `low-risk` areas, not only in `high-risk` zones.

# In[189]:


plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# The heatmap shows how different factors are related to each other in the accident data.

# Accident by Weekend 

# In[163]:


plt.figure()
sns.countplot(x='is_weekend', data=df)
plt.title("Accidents by weekend")
plt.show()


# 1 = weekend (Saturday-Sunday).
# 0 = weekday (Monday to Friday).
# Sabse jyada accident `weekday's` me huye hai 

# Is peak hour

# In[167]:


plt.figure()
sns.countplot(x="is_peak_hour",data=df)
plt.title("Accident during peak hours")
plt.show()


# 1 = True → Peak hour (rush time)
# Usually: morning (8–10 AM) & evening (5–8 PM)
# 0 = False → Non-peak hour
# Sabse jyada accident `Non-peak-hours` me huye hai 

# #### Festival Impact

# In[120]:


plt.figure()
sns.countplot(x='festival', data=df)
plt.title("Accidents During Festivals")
plt.xticks(rotation=45)
plt.show()


# Accident during festival. `Holi` me sabse jyada Accident

# In[173]:


plt.figure()
sns.countplot(x='Acc_Reason', data=df)
plt.title("Accidents reason")
plt.xticks(rotation=45)
plt.show()


# ### Accident reason

# In[183]:


plt.figure(figsize=(7,7))

df['Acc_Reason'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=90)


# This pie chart shows the reason behind accident
# Most of accident have occurred `distraction` and `OverSpeeding

# ## Project Conclusion

# ####  1. High-Risk Conditions Identified
# Accidents are more frequent during:
# Night hours 
# Peak traffic time
# Reduced visibility + driver fatigue = higher risk

# ####  2. Weather Plays a Major Role
# Conditions like:
# Fog 
# Rain 
# Increase accident probability due to:
# Low visibility
# Slippery roads

# ####  3. Human Factors Are Critical
# Major causes observed:
# Over-speeding 
# Drunk driving 
# Distraction 
# Human behavior is a key contributor

# #### 4. Traffic Density Impact
# High traffic density → more accidents
# Congestion increases chances of collisions

# ####  5. Road Type Influence
# Highways & urban roads show more accidents
# Due to:
# High speed (highways)
# Heavy traffic (urban)

# ####  6. Severity Insights
# Most accidents are:
# Minor 
# But:
# Fatal cases increase in high-risk conditions

# ####  7. Festival Impact
# Slight increase in accidents during festivals
# Due to:
# More travel
# Rush & crowd

# ####  8. Risk Score Validation
# Risk score (0–1) effectively represents:
# Probability of accident severity
# Higher score = higher danger
# Final Summary (Short – Interview Ready)

# ## “This project shows that road accidents are mainly influenced by human behavior, weather conditions, and traffic patterns. High-risk situations like night driving, bad weather, and heavy traffic significantly increase both accident frequency and severity.The analysis helps in identifying critical factors and can be used to improve road safety and predictive systems.” 
