library("tidyverse")
library("GGally")

filename = "/Users/malkusch/PowerFolders/LaSCA/thr_alpha/SCTCAPS_thr_complete_noUnits_200821.csv"
df_data = readr::read_csv(file=filename, col_names = TRUE) %>%
  dplyr::select(-c("X1"))

df_data %>%
  dplyr::filter(!((Id == 2) & (day == 1))) %>%
  dplyr::filter(!((Id == 13) & (day == 2))) %>%
  dplyr::filter(time > -15) %>%
  dplyr::filter(frame == 0) %>%
  dplyr::filter(application_type == 'i.c.') %>%
  dplyr::filter(effect != 'allodynia') %>%
  dplyr::select(c("time", "pxlSize_bl", "median_bl", "mean_bl", "std_bl", "thr")) %>%
  GGally::ggpairs(axisLabels = "none") +
  ggplot2::theme_bw() +
  ggplot2::labs(title = "Raw data correlation matrix")

# df_data %>%
#   dplyr::filter(application_type == 'i.c.') %>%
#   dplyr::filter(effect == 'allodynia') %>%
#   #dplyr::select(c("Id", "time", "pxlSize_bl", "median_bl", "mean_bl", "std_bl", "min_bl", "max_bl", "thr")) %>%
#   ggplot2::ggplot() +
#   ggplot2::geom_point(mapping = aes(x = median_bl, y = thr, color = Id))
  

df_data %>%
  dplyr::filter(!((Id == 2) & (day == 1))) %>%
  dplyr::filter(!((Id == 13) & (day == 2))) %>%
  dplyr::filter(frame == 0) %>%
  dplyr::filter(application_type == 'i.c.') %>%
  dplyr::filter(effect == 'allodynia') %>%
  ggplot2::ggplot() +
  ggplot2::geom_boxplot(mapping = aes(x = median_bl, y = thr, group = median_bl)) #+

#
# Analysis of pxlSize
#

df_data %>%
  dplyr::filter(!((Id == 2) & (day == 1))) %>%
  dplyr::filter(!((Id == 13) & (day == 2))) %>%
  dplyr::filter(frame == 0) %>%
  ggplot2::ggplot() +
  geom_histogram(mapping = aes(x=pxlSize))