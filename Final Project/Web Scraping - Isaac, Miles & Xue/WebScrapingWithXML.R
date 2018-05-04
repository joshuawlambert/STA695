# R Web scraping using packages
# http://www.columbia.edu/~cjd11/charles_dimaggio/DIRE/styled-4/styled-6/code-13/


install.packages("XML")
library(XML)

# Specify the URL
URL0=paste("http://www.spotrac.com/nfl/rankings/ ")

# Get the HTML information
d0=htmlParse(URL0,encoding="UTF-8") 

# Get the table
d0.table<- readHTMLTable(d0,stringsAsFactors=F)

# Change the table format from list to data.frame
d0.data=data.frame(d0.table[[1]])






