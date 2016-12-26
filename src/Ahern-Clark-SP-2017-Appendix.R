# # Visualizing and formatting the data

# We load the data into $R$, plot the data and output in a format that can be used to fit data to the functional cycle.

# In[18]:

library(ggplot2, warn.conflicts=F, quietly=T)
library(dplyr, warn.conflicts=F, quietly=T)
library(extrafont)
library(reshape2, quietly=T)
neg.data.full = read.csv("../data/neg-data.csv", header=T)
neg.data.full = tbl_df(neg.data.full)


# Here we filter out texts that are known to be outliers and exclude everything but sentential negation in declaratives.

# In[19]:

excluded.texts = c("CMORM","CMBOETH","CMNTEST","CMOTEST")
#sFilter out tokens without do-support label, year, or type
neg.data = neg.data.full %>%
    filter(finite != "-") %>% # Exclude non-finite clauses
    filter(clausetype != "imperative") %>% # Exclude imperatives
    filter(exclude != "only") %>% # Exclude focus constructions
    filter(exclude != "constituent") %>% # Exclude constituent negation
    filter(exclude != "contraction") %>% # Exclude contraction
    filter(exclude != "coordination") %>% # Exclude coordinated clauses
    filter(exclude != "concord") %>% # Exclude cases of negative concord
    filter(exclude != "X") %>% # Exclude corpus errors
    filter(! author %in% excluded.texts) %>% # Exclude texts
    mutate(stages = ifelse(has.both, 2, ifelse(has.ne, 1, 3))) %>%
    select(year, author, stages)


# We plot the data for all three variants over the course of Middle English.

# In[20]:

neg.plot.data =  neg.data %>% group_by(year, author) %>%
    summarize(total=n(), ne=sum(stages==1, na.rm=TRUE)/total,
              not=sum(stages==3, na.rm=TRUE)/total, ne.not=sum(stages==2, na.rm=TRUE)/total)

neg.plot.data = melt(neg.plot.data, id=c("year", "author", "total"))

p = ggplot(neg.plot.data, aes(x = year, y = value, color = variable)) +
      geom_point(aes(size = total), alpha = 0.5) +
      geom_smooth(method="loess", se = F, size=4) +
      scale_x_continuous(name="Year", limits=c(1100, 1500)) +
      scale_y_continuous(name="Proportion of forms", breaks=seq(0,1,.25)) +
      scale_size_area("N", max_size = 20) +
      theme(text = element_text(size=20, family="Times New Roman"), legend.position="none") +
      coord_cartesian(ylim = c(-.1,1.1))
print(p)
ggsave('../local/out/neg-plot.pdf', height=6, width=8)


# We also plot the data treating post-verbal tokens *as if* they were bipartite tokens in order to model the functional cycle.

# In[21]:

# Compare ne to ne...not and not
first.data = neg.data %>% group_by(year) %>%
    mutate(value = as.integer(! stages==1)) %>%
    select(year, author, value)

first.plot.data = first.data %>%
    group_by(year, author) %>%
    summarize(p = sum(value)/n(), total=n())

p = ggplot(aes(x = year, y = p), data = first.plot.data) +
  geom_point(aes(size = total), alpha = 0.5, position = "identity") +
  geom_smooth(method="loess", se = F, size=4) +
  scale_x_continuous(name="Year", limits=c(1100, 1500)) +
  scale_y_continuous(name="Proportion of forms", breaks=seq(0,1,.25)) +
  theme(text = element_text(size=20, family="Times New Roman")) +
  theme(legend.position="none") +
  scale_size_area("N", max_size = 20) +
  coord_cartesian(xlim = c(1090,1540)) +
  coord_cartesian(ylim = c(-.1,1.1))

print(p)
ggsave('../local/out/func-plot.pdf', height=6, width=8)


# Note that if we were to compare only $ne$ and $ne...not$, excluding $not$ entirely from our analysis, we would run the risk of attributing too much to noisy fluctuations after 1350 when $ne$ and $ne...not$ combined cease to be the majority of forms. Indeed, after 1350 $ne$ becomes more frequent than $ne...not$ again.

# In[22]:

# Compare ne to ne...not and not
exclude.data = neg.data %>%
  filter(! stages == 3) %>%
  group_by(year) %>%
  mutate(value = as.integer(! stages==1)) %>%
  select(year, author, value)

exclude.plot.data = exclude.data %>%
  group_by(year, author) %>%
  summarize(p = sum(value)/n(), total=n())

p = ggplot(aes(x = year, y = p), data = exclude.plot.data) +
  geom_point(aes(size = total), alpha = 0.5, position = "identity") +
  geom_smooth(method="loess", se = F, size=4) +
  scale_x_continuous(name="Year", limits=c(1100, 1500)) +
  scale_y_continuous(name="Proportion of forms", breaks=seq(0,1,.25)) +
  theme(text = element_text(size=20, family="Times New Roman")) +
  theme(legend.position="none") +
  scale_size_area("N", max_size = 20) +
  coord_cartesian(xlim = c(1090,1540)) +
  coord_cartesian(ylim = c(-.1,1.1))

print(p)


# Finally, we output the data in a format that will make it easy to calculate a loss function in a vectorized format.

# In[23]:

functional.cycle.data = data.frame(year=rep(0, 401), has.tokens=rep(0,401), ones=rep(0, 401), zeros=rep(0, 401))
  for (i in c(0:401)) {
    functional.cycle.data$year[i] = i + 1100 - 1
    functional.cycle.data$has.tokens[i] = nrow(first.data %>% filter(year == i + 1100 - 1)) > 0
    functional.cycle.data$ones[i] = nrow(first.data %>% filter(year == i + 1100 - 1, value == 1))
    functional.cycle.data$zeros[i] = nrow(first.data %>% filter(year == i + 1100 - 1, value == 0))
  }
write.csv(functional.cycle.data, "../data/functional-cycle-data.csv", row.names=F)
