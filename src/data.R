library(ggplot2, warn.conflicts=F, quietly=T)
library(dplyr, warn.conflicts=F, quietly=T)
library(extrafont)
library(reshape2, quietly=T)

# Read in the data
neg.data.full = read.csv("../data/neg-data.csv", header=T)
neg.data.full = tbl_df(neg.data.full)
# Filter out texts that are known to be archaic or stilted
excluded.texts = c("CMORM","CMBOETH","CMNTEST","CMOTEST")
# CMBOETH : translation of Boethius' "Consolation of Philosophy", which is notably stilted
# CMORM   : Ormulum is very specific poetic format where adding an additional syllable for the meter is required
# CMOTEST, CMNTEST : Old and new testaments, which are known to carry archaisms longer than other texts
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

# Group data  by year for plotting
neg.plot.data =  neg.data %>% group_by(year, author) %>%
    summarize(total=n(), ne=sum(stages==1, na.rm=TRUE)/total,
              not=sum(stages==3, na.rm=TRUE)/total,
              ne.not=sum(stages==2, na.rm=TRUE)/total)

neg.plot.data = melt(neg.plot.data, id=c("year", 'author', "total"))

ggplot(neg.plot.data, aes(x = year, y = value, color = variable)) +
  geom_point(aes(size = total), alpha = 0.5) +
  geom_smooth(method="loess", se = F, size=4) + # aes(weight = total), span=.6,
  scale_x_continuous(name="Year", limits=c(1100, 1500)) +
  scale_y_continuous(name="Proportion of forms", breaks=seq(0,1,.25)) +   scale_size_area("N", max_size = 20) +
  theme(text = element_text(size=20, family="Times New Roman"), legend.position="none") +
  coord_cartesian(ylim = c(-.1,1.1))
ggsave('../local/out/neg-year-lines.pdf', height=6, width=8)

# Black-and-white printer friendly version
ggplot(neg.plot.data, aes(x = year, y = value, color = variable, linetype = variable)) +
  geom_point(aes(size = total), alpha = 0.5) +
  geom_smooth(method="loess", se = F, size=4) + # aes(weight = total), span=.6,
  scale_x_continuous(name="Year", limits=c(1100, 1500)) +
  scale_y_continuous(name="Proportion of forms", breaks=seq(0,1,.25)) +   scale_size_area("N", max_size = 20) +
  theme_bw() + theme(text = element_text(size=20, family="Times New Roman"), legend.position="none") +
  coord_cartesian(ylim = c(-.1,1.1))
ggsave('../local/out/neg-year-lines-bw.pdf', height=6, width=8)

# What it would look like to compare just ne and ne...not
no.not.data = neg.data %>% group_by(year) %>% filter(stages != 3) %>%
                      mutate(value = as.integer(! stages==1)) %>% select(year, author, value)
no.not.plot.data = no.not.data %>% group_by(year, author) %>%
    summarize(p = sum(value)/n(), total=n())
ggplot(aes(x = year, y = p), data = no.not.plot.data) +
  geom_point(aes(size = total), alpha = 0.5, position = "identity") +
  geom_smooth(method="loess", se = F, size=4) +
  scale_x_continuous(name="Year", limits=c(1100, 1500)) +
  scale_y_continuous(name="Proportion of forms", breaks=seq(0,1,.25)) +
  theme(text = element_text(size=20, family="Times New Roman")) + theme(legend.position="none") +
  scale_size_area("N", max_size = 20) +  coord_cartesian(xlim = c(1090,1540)) + coord_cartesian(ylim = c(-.1,1.1))
ggsave('../local/out/no-not-plot.pdf', height=6, width=8)

# Comparing ne to ne...not and not
func.data = neg.data %>% group_by(year) %>% mutate(value = as.integer(! stages==1)) %>% select(year, author, value)
func.plot.data = func.data %>% group_by(year, author) %>%
    summarize(p = sum(value)/n(), total=n())
ggplot(aes(x = year, y = p), data = func.plot.data) +
  geom_point(aes(size = total), alpha = 0.5, position = "identity") +
  geom_smooth(method="loess", se = F, size=4) +
  scale_x_continuous(name="Year", limits=c(1100, 1500)) +
  scale_y_continuous(name="Proportion of forms", breaks=seq(0,1,.25)) +
  theme(text = element_text(size=20, family="Times New Roman")) + theme(legend.position="none") +
  scale_size_area("N", max_size = 20) +  coord_cartesian(xlim = c(1090,1540)) + coord_cartesian(ylim = c(-.1,1.1))
ggsave('../local/out/func-plot.pdf', height=6, width=8)

# Black-and-white printer friendly version
ggplot(aes(x = year, y = p), data = func.plot.data) +
  geom_point(aes(size = total), alpha = 0.5, position = "identity") +
  geom_smooth(method="loess", se = F, size=4) +
  scale_x_continuous(name="Year", limits=c(1100, 1500)) +
  scale_y_continuous(name="Proportion of forms", breaks=seq(0,1,.25)) +
  theme_bw() + theme(text = element_text(size=20, family="Times New Roman")) + theme(legend.position="none") +
  scale_size_area("N", max_size = 20) +  coord_cartesian(xlim = c(1090,1540)) +
  coord_cartesian(ylim = c(-.1,1.1))
ggsave('../local/out/func-plot-bw.pdf', height=6, width=8)

functional.cycle.data = data.frame(year=rep(0, 401), has.tokens=rep(0,401), ones=rep(0, 401), zeros=rep(0, 401))
for (i in c(0:401)) {
    functional.cycle.data$year[i] = i + 1100 - 1
    functional.cycle.data$has.tokens[i] = nrow(func.data %>% filter(year == i + 1100 - 1)) > 0
    functional.cycle.data$ones[i] = nrow(func.data %>% filter(year == i + 1100 - 1, value == 1))
    functional.cycle.data$zeros[i] = nrow(func.data %>% filter(year == i + 1100 - 1, value == 0))
}
write.csv(functional.cycle.data, "../data/functional-cycle-data.csv", row.names=F)
