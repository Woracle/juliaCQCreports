# The purpose of this project is to explore the use of
# Julia for Analysis and NLP tasks.
# This will be about exploring capabilities and the workflow. 
# as well as usability from a first time julia users perspective.

# Hounestly the data selection, processing and methods used
# probably make no sense. (as i said its for the learning)

# Lets start with getting the packages we want
using Pkg
#Pkg.add("HTTP")
#Pkg.add("DataFrames")
#Pkg.add("JSON")
#Pkg.add("DataStructures")
#Pkg.add("PyCall") 
#Pkg.add("TextAnalysis")

using HTTP
using DataFrames
using JSON
using DataStructures
using TextAnalysis
using PyCall
# To be more interesting than the standard download text 
# corpus and test methods we are going to use data from 
# a public API.
# This allows us to test interaction with REST APIs. HTTP 
# requests and accessing data within nested structures.

# the CQC API allows us to gather information on locations
# the URL is https://api.cqc.org.uk/public/v1/locations
request = HTTP.request("GET", "https://api.cqc.org.uk/public/v1/locations")
request.status
json = JSON.Parser.parse(String(request.body))

# Extract the Location data from the json
vectors = json.vals[4]

# Combine Vectors into a Dataframe
# the dot (broadcast) after Dataframe in Dataframe.
# means the dataframe function gets called on
# each item in the array. Super neat way to "vectorise"
# a function.
df = vcat(DataFrame.(vectors)...)

# View the DataFrame
display(df)

# create a function to query CQC API based
# unneccesary but worth testing our ability to
# create new function. 
# Also worth noting CQC API is IMO annoying
# Notes: scoping for try catch seems odd.
# one solution involves pushing everything into a
# function to handle scoping. (imo not the end of the world.)
# the other is assinging the try to a variable. 
function CQC_Loc_Details_API(locid::String)

    req_string = "https://api.cqc.org.uk/public/v1/locations/" * locid
    request = HTTP.request("GET", req_string)
        if request.status != 200
            error("requst status not 200")
        end
    json = JSON.Parser.parse(String(request.body))

    # Extract key information about the service
    info_List = ["name", "registrationDate", "inspectionDirectorate", "numberOfBeds", "careHome"]
    info_Dict = OrderedDict{String, Any}()

    for col in info_List
        info_Dict[col]  =  try
        json[col]
        catch;
         Any[]
        end
    end

    info_df = DataFrame(info_Dict)

    # get current overall rating 
    # scoping
    current_rating =    try 
         json["currentRatings"]["overall"]["rating"]
           catch;
         "Not Currently Rated"
        end
    rat_df = DataFrame(currentRating = current_rating)


    # From the Json pull the report information

    reports_df =  try 
         vcat(DataFrame.(json["reports"])...)
    catch;
     DataFrame(linkId = nothing, 
        reportType = nothing, 
        firstVisitDate = nothing,
        reportUri = nothing,
        reportDate = nothing)
    end

    # Merge the dataframes into a record
    df_out = hcat(
        repeat( hcat(info_df, rat_df) , inner = nrow(reports_df)), 
        reports_df)

    insertcols!(df_out, 1, :locationId => locid)

    return(df_out)
end

# We create this function to extract information from a service.
# natively we can use broadcast to "vectorise" our custom function.
# no need for loops or map...
# i cannot stress how neat this is.
info = CQC_Loc_Details_API.(df.locationId[1:10])
API_data = vcat(DataFrame.(info)...)

# Next step is to use information in the linkid column
# in a new function to get the inspection reports text
# this will be the free text we use for our analysis.

function GET_CQC_Report_Text_API(linkId)
        try
        req_string = "https://api.cqc.org.uk/public/v1/reports/"*linkId

        request = HTTP.request("GET", req_string, ["Accept" => "text/plain"]) # , ["Accept" => "text/plain"]
            if request.status != 200
                error("requst status not 200")
            end

        return(request)

        catch
        return("API unsuccessful")
        end
end

# we will now use this function to add the report text to our dataframe
report_text = GET_CQC_Report_Text_API.(API_data.linkId)
rpt_txt = vcat(String.(report_text)...) # i'm not 100% sure this is necessary but it worked
API_data = hcat(API_data, rpt_txt)
rename!(API_data,:x1 => :report_text) # for renaming column ! means we make the change in place.

# CQC's API returns the content of the report as text but it also includes info as part of
# the HTTP response. 
# lets generate a regex to remove the offending string
http_response_regex = r"HTTP.*GMT"s # syntax is standard (at least with R) 
# with only these extra prefixes and suffixes to control some behaviour
# r creates the regex object and s means we ignore \n \r \t etc.
# which is nice. 
# replace text in place using broadcast
API_data.report_text = replace.(API_data.report_text, http_response_regex => "")

# The textAnalytics package provides a number of utilities for working with 
# text in julia. The first is stringDocuments as a way of holding corpuses of text.

# Corpus allows us to group string documents. We are only interested in services with reports
# so lets filter out those which were unsucessful.
crps = Corpus(StringDocument.(API_data[API_data[: , :report_text] .!= "API unsuccessful", :report_text]))

# now that we have a corpus we can use the framework set up
# to quickly perform multiple cleaning steps. 
prepare!.(crps, strip_corrupt_utf8 | strip_case | strip_html_tags)

# worth exploring in the future how we might extend this flags approach to
# include personal cleaning flags for convienience... 

# Anyway now that we have "clean" (it's not clean) data we can start
# explore some of the current package offering for NLP.

# lets first look at creating a Document Term Matrix and td_idf
# the text analytics package offers three formats of dtm 
# The first is a special object class
update_lexicon!(crps) # updates the lexicon inplace
dm = DocumentTermMatrix(crps) # an assumption is this might not
# always work nicely with other packages so the next two allow you create generic
# Matrices which can be used by other functions
# to create a sparse Matrix
sparse_dtm = dtm(dm)
# or a dense one
dense_dtm = dtm(dm, :dense)

# we can again simply generate td_idf with a simple function call. 
invrs_doc_freq = tf_idf(dm)

# These are simple approaches to converting text data into a numeric representation 
# of those documents which can be inputs for other statistical and ML techniques (classifiers, similiarity etc)
# Modern techniques allow us to generate more meaningful representations of text. 

# for example word embeddings and more specifically word2vec uses a shallow Neural Network
# to place words within a n-dimentional space. with the relative position of 
# words conveying something meaningful about those words. 

# word2vec.jl and embeddings.jl packages doesn't have in my view a statisfactory approach to 
# training on in memory documents... probably need to do a bit more exploration. 

# also there doesn't seem to be a straight forward implimentation of 
# doc2vec (maybe this could be a nice next julia project for myself).

# instead we can explore the usage of python from within Julia, to close gaps where
# packages don't currently exist in julia
pyimport_conda("gensim", "gensim")
gensim = pyimport("gensim")
Doc2Vec = pyimport("gensim.models.doc2vec")

# first lets create a python object. which allows us to convert
# a DataFrame into a Tagged Document object something we need to pass
# into the gensim module
py_df = PyObject(API_data)
py"""
class df_2_taggedDocument(object):
    def __init__(self, source_df, text_col, tag_col):
        self.source_df = source_df # this is the original dataframe
        self.text_col = text_col # This is the column with the free text
        self.tag_col = tag_col # This is the tag/documentID. 

    def __iter__(self):
        for i, row in self.source_df.iterrows():
            yield Doc2Vec.TaggedDocument(words=gensim.utils.simple_preprocess(row[self.text_col]), 
                                 tags=[row[self.tag_col]])                       
"""
# we now have our tagged doc item 
tagged_doc = py"df_2_taggedDocument"(source_df = py_df, text_col = "report_text", tag_col = "linkId")

# we can then pass this through python's gensim's Doc2Vec modules
# to get document vectors for all our documents.
model = Doc2Vec.Doc2Vec(size= 100, min_count = 5, dm = 1)

model.build_vocab(tagged_doc)
# for some reason we are running into an issue where model seems to be missing
# iterrows for build_vocab... My thoughts are that this might be a version issue
# however, hard to diagnose. 
model.train(tagged_doc, epochs = 10) # this wont work without the vocab above

# testing with the same set-up for a very simple example also creates an error. 
a = Doc2Vec.TaggedDocument(words=gensim.utils.simple_preprocess("This is some text"), tags = ["tag01"])
b = Doc2Vec.TaggedDocument(words=gensim.utils.simple_preprocess("This is some other great text"), tags = ["tag02"])

tnames = (:words, :tags);

c = (;zip(tnames, a)...)

model.build_vocab(c)
# So it looks like when the PyObject is being being stored. The named elements (i.e. words and tags are being lost)
# There is also an issue with conversion from python namedTuple > Julia namedTuple > python namedTuple which means
# we lose the names. https://github.com/JuliaPy/PyCall.jl/issues/175 

# A potential Solution might be to ensure the outputs from the TaggedDocument function
# remain as a PyObject. So no conversion mangling doesn't occur.

