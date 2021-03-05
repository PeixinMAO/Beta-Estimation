library(shiny)
library(shinydashboard)
library(plotly)
library(magrittr)

#Change this dir to the root of the repository
parentDir <- "/home/khll/School/ENSAE/CIAM/main"
dataDir = paste(parentDir, "output", sep = '/')


data <-
    list.files(dataDir, pattern = "*.csv", full.names = TRUE) %>%
    lapply(read.csv)
names(data) <- list.files(dataDir, pattern = "*.csv") %>%
    gsub(pattern = ".{4}$", replacement = '')

data %<>% lapply(function(x) {
    x$Date <- as.Date(x$Date, format = "%Y-%m-%d")
    return(x)
})

minDate <- as.Date(max(sapply(data,function(x) {min(x$Date)})),origin = "1970-01-01")
maxDate <- as.Date(min(sapply(data,function(x) {max(x$Date)})),origin = "1970-01-01")

data %<>% lapply(function(x) {
    res <- x[which(x$Date >= minDate & x$Date <= maxDate),]
    return(res)
})


ui <- dashboardPage(
    dashboardHeader(),
    dashboardSidebar(
        selectInput("vars", "Pick variable(s)", names(data), multiple = T),
        uiOutput("selectStock"),
        uiOutput(("selectDate"))
    ),
    dashboardBody(plotlyOutput("plot"))
)

server <- function(input, output) {
    output$selectStock <- renderUI({
        req(input$vars)
        selectInput("stock", "Pick a stock", colnames(data[[input$vars[1]]])[-1])
    })
    
    output$selectDate <- renderUI({
        dateRangeInput(
            'dateRange',
            label = 'Select date range',
            startview = "year",
            min = minDate,
            start = minDate,
            max = maxDate,
            end = maxDate
        )
    })
    
    df.data <- reactive({
        req(input$vars)
        req(input$stock)
        req(input$dateRange)
        tmp <- sapply(c(input$vars), function(x) {
            data[[x]][input$stock]
        }) %>% as.data.frame() %>% cbind.data.frame("Date" = data[[1]]$Date)
        tmp <-
            tmp[which(tmp$Date >= input$dateRange[1] &
                          tmp$Date <= input$dateRange[2]), ]
        return(tmp)
    })
    
    
    output$plot <-
        renderPlotly({
            p <- plot_ly(type = "scatter", mode = "lines") %>%
                layout(xaxis = list(title = "Date"),
                       yaxis = list(title = "Beta"))
            for (trace in setdiff(colnames(df.data()), "Date")) {
                p %<>% add_trace(
                    x = df.data()[["Date"]],
                    y = df.data()[[trace]],
                    name = trace,
                    type = 'scatter',
                    mode = 'lines'
                )
            }
            return(p)
        })
    
}

shinyApp(ui, server)
