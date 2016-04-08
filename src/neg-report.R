##########################################################################
# Load libraries and scripts
library(ggplot2)
library(plyr)
library(dplyr)
library(reshape2)
library(xtable)
source("neg-data.R")
source("binning.R")
##########################################################################
# Read in data file and filter out texts and contexts
neg.data.full = cleanNegData("coding.cod.ooo")
neg.data.full = tbl_df(neg.data.full)
# Filter out tokens without do-support label, year, or type
excluded.texts = c("CMBOETH", "CMORM", "CMNTEST","CMOTEST")
# CMBOETH : translation of Boethius' "Consolation of Philosophy", which is notably stilted
# CMORM   : Ormulum is very specific poetic format
# CMOTEST, CMNTEST : Old and new testaments
# Exclude tokens where 'ne' is contracted, appears in negative concord, or looks like predicate negation
neg.data.full = neg.data.full %>% filter(! author %in% excluded.texts & exclude != "X") 
# Plot individual documents
neg.plot.auth = neg.data.full %>% group_by(year) %>% summarize(total=n(), ne=sum(neg.type=="ne", na.rm=TRUE)/total,not=sum(neg.type=="not", na.rm=TRUE)/total,ne.not=sum(neg.type=="both", na.rm=TRUE)/total)
neg.plot.auth = melt(test, id=c("year", "total"))
# Plot points and smooth fits
ggplot(aes(x = year, y = value, color = variable), data = neg.plot.auth) +
  geom_point(aes(size = total), alpha = 0.5, position = position_jitter()) +
  geom_smooth(aes(weight = total),method="loess", span=.5, se = FALSE, size=4) + # span=.5,
  xlab("Year") +   ylab("Proportion forms") +   scale_size_area("N", max_size = 20) +
  theme(text = element_text(size=30)) +   theme(legend.position="none") +
  coord_cartesian(xlim = c(1090,1540)) +  coord_cartesian(ylim = c(-.1,1.1))
###############################
# Lump things together
###############################
# Compare ne to ne...not and not
first.data = neg.data.full %.% group_by(year) %>% mutate(value = ! stages=="1") %>% select(year, value)
# Plot comparison
first.plot.data = first.data %.% group_by(year) %.% summarize(count = sum(as.numeric(value)), total=n()) %.% mutate(p = count / total)
ggplot(aes(x = year, y = p), data = first.plot.data) +
  geom_point(aes(size = total), alpha = 0.5, position = position_jitter()) +
  geom_smooth(aes(weight = total),method="loess",span=.5,  se = FALSE, size=5) + # span=.5,
  xlab("Year") +   ylab("Proportion") + theme(text = element_text(size=20)) + theme(legend.position="none") +
  scale_size_area("N", max_size = 20) +  coord_cartesian(xlim = c(1090,1540)) + coord_cartesian(ylim = c(-.1,1.1))
# Run FIT
print(xtable(q.table(first.data, 8), digits=c(0,0,0,0,0,4,4,4,0,0,4)), include.rownames=FALSE) # These are our main results
do.FIT(do.by.quantile(first.data, 9))[c(7,10)]  # Breaks are non-unique for finer bins
################################
# Compare ne and ne...not to not
second.data = neg.data.full %>% group_by(year) %.% mutate(value = stages=="3") %>% select(year, value)
# Plot comparison
second.plot.data = second.data %.% group_by(year) %.% summarize(count = sum(as.numeric(value)), total=n()) %.% mutate(p = count / total)
ggplot(aes(x = year, y = p), data = second.plot.data) +
  geom_point(aes(size = total), alpha = 0.5, position = position_jitter()) +
  geom_smooth(aes(weight = total),method="loess",span=.5, se = FALSE, size=5) + # span=.5
  xlab("Year") + ylab("Proportion") + theme(text = element_text(size=20)) + theme(legend.position="none") +
  scale_size_area("N", max_size = 20) + coord_cartesian(xlim = c(1090,1540)) + coord_cartesian(ylim = c(-.1,1.1))
# Run FIT
print(xtable(q.table(second.data, 8), digits=c(0,0,0,0,0,4,4,4,0,0,4)), include.rownames=FALSE) # These are our main results
do.FIT(do.by.quantile(second.data, 9))[c(7,10)]  # Non-unique breaks or absorption events
###############################
# Split things by date ~1350
###############################
# Compare ne to ne...not before ~1350
first.data = neg.data.full %.% filter(stages !=3 & year <=1350) %.% group_by(year) %.% mutate(value = ! stages=="1") %>% select(year, value)
# Plot comparison
first.plot.data = first.data %.% group_by(year) %.% summarize(count = sum(as.numeric(value)), total=n()) %.% mutate(p = count / total)
ggplot(aes(x = year, y = p), data = first.plot.data) +
  geom_point(aes(size = total), alpha = 0.5, position = position_jitter()) +
  geom_smooth(aes(weight = total),method="loess",  se = FALSE, size=5) + # span=.5,
  xlab("Year") + ylab("Proportion") +   theme(text = element_text(size=20)) +   theme(legend.position="none") +
  scale_size_area("N", max_size = 20) + coord_cartesian(xlim = c(1090,1540)) + coord_cartesian(ylim = c(-.1,1.1))
# Run FIT
print(xtable(q.table(first.data, 5), digits=c(0,0,0,0,0,4,4,4,0,0,4)), include.rownames=FALSE) # These are our main results
do.FIT(do.by.quantile(first.data, 6))[c(7,10)] # Non-unique breaks
# Compare ne...not to not after ~1350
second.data = neg.data.full %.% filter(stages != 1 & year >= 1350) %.% group_by(year) %.% mutate(value = stages=="3") %>% select(year, value)
# Plot comparison
second.plot.data = second.data %.% group_by(year) %.% summarize(count = sum(as.numeric(value)), total=n()) %.% mutate(p = count / total)
ggplot(aes(x = year, y = p), data = second.plot.data) +
  geom_point(aes(size = total), alpha = 0.5, position = position_jitter()) +
  geom_smooth(aes(weight = total),method="loess", span=.45, se = FALSE, size=5) +
  xlab("Year") + ylab("Proportion") + theme(text = element_text(size=20)) + theme(legend.position="none") +
  scale_size_area("N", max_size = 20) + coord_cartesian(xlim = c(1090,1540)) + coord_cartesian(ylim = c(-.1,1.1))
# Run FIT
print(xtable(q.table(second.data, 5), digits=c(0,0,0,0,0,4,4,4,0,0,4)), include.rownames=FALSE) # These are our main results
do.FIT(do.by.quantile(second.data, 6))[c(7,10)] # Non-unique breaks
######################################
# Plot the number of documents by year
auth.plot.data = neg.data.full %>% group_by(author, year) %>% summarize() %>% arrange(year)
ggplot(auth.plot.data, aes(x=year)) + geom_histogram(binwidth=1) + theme(text = element_text(size=20))
# Add noise to year for each document
set.seed(1)
neg.data.jitter = neg.data.full %>% group_by(author) %>% mutate(year = year + (year[1] - jitter(year[1], amount=.5)))
# Check to make sure number of unique dates matches documents
length(unique(neg.data.jitter$year))
length(unique(neg.data.jitter$author))
########################################
# Disaggreagate and Lump things together
########################################
# Compare ne to ne...not and not
first.data = neg.data.jitter %.% group_by(year, author) %.% mutate(value = ! stages=="1") %>% mutate(jyear = jitter(year, amount=.5)) %>% select(year, value)
# Plot comparison
first.plot.data = first.data %.% group_by(year) %.% summarize(count = sum(as.numeric(value)), total=n()) %.% mutate(p = count / total)
ggplot(aes(x = year, y = p), data = first.plot.data) +
  geom_point(aes(size = total), alpha = 0.5, position = position_jitter()) +
  geom_smooth(aes(weight = total),method="loess", span=.5, se = FALSE, size=5) +
  xlab("Year") + ylab("Proportion") + theme(text = element_text(size=20)) + theme(legend.position="none") +
  scale_size_area("N", max_size = 20) + coord_cartesian(xlim = c(1090,1540)) + coord_cartesian(ylim = c(-.1,1.1))
# Run FIT
print(xtable(q.table(first.data, 10), digits=c(0,0,0,0,0,4,4,4,0,0,4)), include.rownames=FALSE) # These are our main results
do.FIT(do.by.quantile(first.data, 11)) # Absorption event
do.FIT(do.by.quantile(first.data, 12)) # 0.2835811, 0.1470875
do.FIT(do.by.quantile(first.data, 13)) # 0.1361334, 0.3205971
do.FIT(do.by.quantile(first.data, 14)) # 
do.FIT(do.by.quantile(first.data, 15)) # 
do.FIT(do.by.quantile(first.data, 16)) # Absorption event
# Compare ne and ne...not to not
second.data = neg.data.jitter %>% group_by(year) %.% mutate(value = stages=="3") %>% mutate(jyear = jitter(year, amount=.5)) %>% select(year, value)
# Plot comparison
second.plot.data = second.data %.% group_by(year) %.% summarize(count = sum(as.numeric(value)), total=n()) %.% mutate(p = count / total)
ggplot(aes(x = year, y = p), data = second.plot.data) +
  geom_point(aes(size = total), alpha = 0.5, position = position_jitter()) +
  geom_smooth(aes(weight = total),method="loess", span=.45, se = FALSE, size=5) +
  xlab("Year") + ylab("Proportion") + theme(text = element_text(size=20)) + theme(legend.position="none") +
  scale_size_area("N", max_size = 20) + coord_cartesian(xlim = c(1090,1540)) + coord_cartesian(ylim = c(-.1,1.1))
# Run FIT
print(xtable(q.table(second.data, 10), digits=c(0,0,0,0,0,4,4,4,0,0,4)), include.rownames=FALSE) # These are our main results
do.FIT(do.by.quantile(second.data, 11)) #
do.FIT(do.by.quantile(second.data, 12)) #
do.FIT(do.by.quantile(second.data, 13)) #
do.FIT(do.by.quantile(second.data, 14)) #
do.FIT(do.by.quantile(second.data, 15)) #

###################################################
# Disaggreagate and Split things by date ~1350
###################################################
# Compare ne to ne...not before ~1350
first.data = neg.data.jitter %.% filter(stages !=3 & year <=1351) %.% group_by(year) %.% mutate(value = ! stages=="1") %>% select(year, value)
# Plot comparison
first.plot.data = first.data %.% group_by(year) %.% summarize(count = sum(as.numeric(value)), total=n()) %.% mutate(p = count / total)
ggplot(aes(x = year, y = p), data = first.plot.data) +
  geom_point(aes(size = total), alpha = 0.5, position = position_jitter()) +
  geom_smooth(aes(weight = total),method="loess", se = FALSE, size=5) +
  xlab("Year") + ylab("Proportion") +   theme(text = element_text(size=20)) +   theme(legend.position="none") +
  scale_size_area("N", max_size = 20) + coord_cartesian(xlim = c(1090,1540)) + coord_cartesian(ylim = c(-.1,1.1))
# Run FIT
print(xtable(q.table(first.data, 5), digits=c(0,0,0,0,0,4,4,4,0,0,4)), include.rownames=FALSE) # These are our main results
do.FIT(do.by.quantile(first.data, 4))[c(7,10)] # Boundary  p = .13394961
do.FIT(do.by.quantile(first.data, 5))[c(7,10)] # Nope      p = .12035848 ! SW-p = 0.02479866
do.FIT(do.by.quantile(first.data, 6))[c(7,10)] # Non-unique breaks
do.FIT(do.by.quantile(first.data, 7))[c(7,10)] # Non-unique breaks
do.FIT(do.by.quantile(first.data, 8))[c(7,10)] # Non-unique breaks
do.FIT(do.by.quantile(first.data, 9))[c(7,10)] # Non-unique breaks
# Compare ne...not to not after ~1350
second.data = neg.data.jitter %.% filter(stages != 1 & year >= 1349) %.% group_by(year) %.% mutate(value = stages=="3")
# Plot comparison
second.plot.data = second.data %.% group_by(year) %.% summarize(count = sum(as.numeric(value)), total=n()) %.% mutate(p = count / total)
ggplot(aes(x = year, y = p), data = second.plot.data) +
  geom_point(aes(size = total), alpha = 0.5, position = position_jitter()) +
  geom_smooth(aes(weight = total),method="loess",  se = FALSE, size=5) +
  xlab("Year") + ylab("Proportion") + theme(text = element_text(size=20)) + theme(legend.position="none") +
  scale_size_area("N", max_size = 20) + coord_cartesian(xlim = c(1090,1540)) + coord_cartesian(ylim = c(-.1,1.1))
# Run FIT
print(xtable(q.table(second.data, 9), digits=c(0,0,0,0,0,4,4,4,0,0,4)), include.rownames=FALSE) # These are our main results
do.FIT(do.by.quantile(second.data, 6)) # Absorption


###############################
# For the appendix
###############################
# Split things apart completely
###############################
# Compare ne to ne...not
first.data = neg.data.full %.% filter(stages !=3) %.% dplyr::group_by(year) %.% dplyr::mutate(value = ! stages=="1")
# Plot comparison
first.plot.data = first.data %.% group_by(year) %.% dplyr::summarize(count = sum(as.numeric(value)), total=n()) %.% dplyr::mutate(p = count / total)
ggplot(aes(x = year, y = p), data = first.plot.data) +
  geom_point(aes(size = total), alpha = 0.5, position = position_jitter()) +
  geom_smooth(aes(weight = total),method="loess", se = FALSE, size=5) + # span=.45, 
  xlab("Year") +
  ylab("Proportion") +
  scale_size_area("N", max_size = 20) +
  theme(text = element_text(size=20)) + 
  theme(legend.position="none") +
  coord_cartesian(xlim = c(1090,1540)) +
  coord_cartesian(ylim = c(-.1,1.1))
# Run FIT
do.FIT(do.by.quantile(first.data, 4))[c(7,10)] # Nope p = .19137741
do.FIT(do.by.quantile(first.data, 5))[c(7,10)] # Nope p = .509845867
do.FIT(do.by.quantile(first.data, 6))[c(7,10)] # Nope p = .4876574
do.FIT(do.by.quantile(first.data, 7))[c(7,10)] # Non-unique breaks
do.FIT(do.by.quantile(first.data, 8))[c(7,10)] # Non-unique breaks
do.FIT(do.by.quantile(first.data, 9))[c(7,10)] # Non-unique breaks
# Compare ne and ne...not to not
second.data = neg.data.full %.% filter(stages != 1) %.% dplyr::group_by(year) %.% dplyr::mutate(value = stages=="3") %>% mutate(jyear = jitter(year, amount=.5))
# Plot comparison
second.plot.data = second.data %.% group_by(year) %.% dplyr::summarize(count = sum(as.numeric(value)), total=n()) %.% dplyr::mutate(p = count / total)
ggplot(aes(x = year, y = p), data = second.plot.data) +
  geom_point(aes(size = total), alpha = 0.5, position = position_jitter()) +
  geom_smooth(aes(weight = total),method="loess", se = FALSE, size=5) + # span=.45, 
  xlab("Year") +
  ylab("Proportion") +
  scale_size_area("N", max_size = 20) +
  theme(text = element_text(size=20)) + 
  theme(legend.position="none") +
  coord_cartesian(xlim = c(1090,1540)) +
  coord_cartesian(ylim = c(-.1,1.1))
# Run FIT
second.data$year = second.data$jyear
do.FIT(do.by.quantile(second.data, 4))[c(7,10)] # Marginal   p = .05787163
do.FIT(do.by.quantile(second.data, 5))[c(7,10)] # Absorption
do.FIT(do.by.quantile(second.data, 6))[c(7,10)] # Nope       p = .15152610
do.FIT(do.by.quantile(second.data, 7))[c(7,10)] # Non-unique breaks
do.FIT(do.by.quantile(second.data, 8))[c(7,10)] # Absorption
do.FIT(do.by.quantile(second.data, 9))[c(7,10)] # Non-unique breaks
###############################
# Lump things around ~1350
###############################
# Compare ne to ne...not and not before ~1350
first.data = neg.data.full %.% filter(year <=1350) %.% dplyr::group_by(year) %.% dplyr::mutate(value = ! stages=="1") %>% select(year, value)
# Plot comparison
first.plot.data = first.data %.% group_by(year) %.% dplyr::summarize(count = sum(as.numeric(value)), total=n()) %.% dplyr::mutate(p = count / total) %.% select(year, p)
ggplot(aes(x = year, y = p), data = first.plot.data, position = position_jitter()) +
  geom_point(aes(size = p), alpha = 0.5) +
  geom_smooth(aes(weight = p),method="loess", span=.45, se = FALSE, size=5) +
  xlab("Year") +
  ylab("Proportion") +
  scale_size_area("N", max_size = 20) +
  theme(text = element_text(size=20)) + 
  theme(legend.position="none") +
  coord_cartesian(xlim = c(1090,1540)) +
  coord_cartesian(ylim = c(-.1,1.1))
# Run FIT
do.FIT(do.by.quantile(first.data, 4))[c(7,10)] # Boundary  p = .13280303
do.FIT(do.by.quantile(first.data, 5))[c(7,10)] # Non-unique breaks
do.FIT(do.by.quantile(first.data, 6))[c(7,10)] # Non-unique breaks
do.FIT(do.by.quantile(first.data, 7))[c(7,10)] # Non-unique breaks
do.FIT(do.by.quantile(first.data, 8))[c(7,10)] # Non-unique breaks
do.FIT(do.by.quantile(first.data, 9))[c(7,10)] # Non-unique breaks
# Compare ne...not to not after ~1350
second.data = neg.data.full %.% filter(year >= 1350) %.% dplyr::group_by(year) %.% dplyr::mutate(value = stages=="3") %>% select(year, value)
# Plot comparison
second.plot.data = second.data %.% group_by(year) %.% dplyr::summarize(count = sum(as.numeric(value)), total=n()) %.% dplyr::mutate(p = count / total) %.% select(year, p)
ggplot(aes(x = year, y = p), data = second.plot.data) +
  geom_point(aes(size = p), alpha = 0.5, position = position_jitter()) +
  geom_smooth(aes(weight = p),method="loess", span=.45, se = FALSE, size=5) +
  xlab("Year") +
  ylab("Proportion") +
  scale_size_area("N", max_size = 20) +
  theme(text = element_text(size=20)) + 
  theme(legend.position="none") +
  coord_cartesian(xlim = c(1090,1540)) +
  coord_cartesian(ylim = c(-.1,1.1))
# Run FIT
do.FIT(do.by.quantile(second.data, 4))[c(7,10)] # Boundary p = .09217675
do.FIT(do.by.quantile(second.data, 5))[c(7,10)] # Marginal p = .05872777
do.FIT(do.by.quantile(second.data, 6))[c(7,10)] # Non-unique breaks
do.FIT(do.by.quantile(second.data, 7))[c(7,10)] # Absorption
do.FIT(do.by.quantile(second.data, 8))[c(7,10)] # Non-unique breaks
do.FIT(do.by.quantile(second.data, 9))[c(7,10)] # Non-unique breaks
################################################################
# Some dates appeared to be wrong! But now are fixed
# Priority of circa data over ante date, unless ? info
################################################################
# What are the author dates?
author.dates = data.frame(auth=character(), year=numeric())
for (a in unique(neg.data.full$author)){
  date = unique(filter(neg.data.full, author == a)$year)
  auth.add = data.frame(auth=a, year=date)  
  author.dates = rbind(author.dates, auth.add)
}
author.dates = arrange(author.dates, year)
write.csv(author.dates, file="PPCME2dates.csv", row.names=FALSE)
# Let's look at the proprotions for a given author
inspect.author = function(a){
  return(neg.data.full %.% dplyr::filter(author == a) %.% dplyr::summarize(ne = sum(stages == "1"), ne.not = sum(stages == "2"), not = sum(stages == "3"), count = n()) %.% dplyr::mutate(ne = ne / count) %.% dplyr::mutate(ne.not = ne.not / count) %.% dplyr::mutate(not = not / count))
}