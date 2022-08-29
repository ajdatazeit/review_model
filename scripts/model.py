import re
import pickle
import unicodedata
from tqdm import tqdm
from datetime import datetime, timezone
import pytz
import numpy as np
import pandas as pd
import simplemma
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

##downloading dictionaries
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

##initializing basic elements
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
tmz=pytz.timezone('Europe/Berlin')

tqdm.pandas(desc="Example Desc")

class Review_Model:
    def __init__(self, **kwargs):
        self.df = kwargs.get("records")
        self.stopwords = self.get_stopwords()
        self.tags_dict = pickle.load(open("../src/tags_dict.pickle","rb"))
        self.tags_dict = self.get_tags_dict(self.tags_dict)
        self.categories = pd.read_csv("../src/final_review_model_sheet.csv")
        self.lang = "en"
        self.get_classes_table()
        self.concept_dict = self.get_near_concept_dict()
        self.class_dict = self.get_keyword_dict()
        nltk.download('wordnet')
        nltk.download('omw-1.4')


    def get_stopwords(self):
        my_file = open(f"../src/review_model_normalization_stopwords.txt", "r")
        stopwords = my_file.read().split("\n")
        stopwords = {word: 1 for word in stopwords}
        return stopwords

    def data_cleaning(self,text:str):
        """to clean the text removing unwanted character and converting 
        the different language character in latin character"""
        text = unicodedata.normalize("NFD",text)
        text = re.sub("[\u0300-\u036f]", "", text).lower()
        text = re.sub(r'[^a-z-\'0-9 ]', '', text)
        text = text.replace("ß","ss")
        text = re.sub(r'-', ' ', text)
        text = text.strip()
        return text

    def get_numbers(self,text):
        numbers= re.findall(r'\d', text)
        numbers.sort()
        numbers="".join(numbers)
        return numbers

    def clean_tags_remove_unigrams(self,tag):
        """ to clean the unigrams from the tags if that unigram is already 
        include in bigrams of any ngram
        @params tags: list of tags
        """
        unigrams = [gram for gram in tag if len(gram.split())<2]
        ngrams = [gram for gram in tag if len(gram.split())>1]
        for unigram in unigrams:
            flag=True
            for ngram in ngrams:
                if unigram in ngram:
                    flag = False
                    break
            if flag == True:
                ngrams.append(unigram)
        return ngrams

    def _get_clean_tags_dict(self,tags_dict):
        """only using the  longest length tag in the dictionary"""
        final_tags_dict = {}
        for key,value in tags_dict.items():
            value = list(sorted(value, key = len))
            final_tags_dict.update({key:value[0]})
        return final_tags_dict

        
    def get_tags_dict(self,tags_dict):
        """ normalizing the tags so that tags like "long lasting" and "last long"
        are treated as one not different
        @params: iterable datatype of tags"""
        print("length of input tags",len(tags_dict))
        updated_tags_dict = {}
        for tag in tags_dict:
            tag = self.data_cleaning(tag)
            ###tag = [simplemma.lemmatize(word,lang=lang).strip() for word in tag.split() if word not in normalized_stopwords]
            tag = [lemmatizer.lemmatize(word).strip() for word in tag.split() if word not in self.stopwords]
            tag.sort()
            tag = " ".join(tag)
            if tag != "":
                updated_tags_dict.update({tag:1})
        print("length of output tag" , len(updated_tags_dict))
        return updated_tags_dict

    def updated_get_n_gramlist(self,spl):
        """ This is the main logic to fing the tags based on ngrams
        so that the time complexity will be minimum """ 
        nngramlist=[]
        test={}
        for n in range(1,6):
            for s in ngrams(spl,n=n):
                ##normalizing the reviews
                ngrams_ = [simplemma.lemmatize(word,lang=self.lang).strip() for word in s if word not in self.stopwords]
                # ngrams_ = [lemmatizer.lemmatize(word).strip()  for word in s if word not in self.stopwords]
                show_gram = " ".join(ngrams_)
                ngrams_.sort()
                ngrams_ = " ".join(ngrams_)   
                if ngrams_ not in test:
                    nngramlist.append([ngrams_,show_gram]) 
                    test.update({ngrams_:1})
        return nngramlist


    def get_tags_for_reviews(self,review):
        """iterating over all the reviews and fing the tag using the ngram logic"""
        temp_tag_list = []
        review = self.data_cleaning(review)
        spl = review.split()
        ngram_list = self.updated_get_n_gramlist(spl)
        for ngrams_ in ngram_list:
            if ngrams_[0] in self.tags_dict:
                temp_tag_list.append(ngrams_[1])
        if temp_tag_list == []:
            return np.nan
        return temp_tag_list

    def Normalize(self):
        cluster={}
        print("length of input data",len(self.tags_dict))
        for tag,freq in tqdm(self.tags_dict.items()):
            tag = self.data_cleaning(tag)
            ##preprocessing operations to normalize the sentenc
            words=tag.split()
            ## normalizing the sentence throw lemmatization and removing stopwords
            words=[simplemma.lemmatize(word,lang=self.lang).strip() for word in words if word not in self.stopwords]
            ##sorting the words so that their order become undifferentiable ["cream","face" ]  ["face","cream"] .
            words.sort()
            key = " ".join(words)
            ##adding the sv so that clustered keyword must have same sv
            if key not in cluster:
                cluster.update({key:{tag:freq}})  
            else:
                cluster[key].update({tag:freq}) 
        return cluster


    def get_mapping_dict(self,cluster):
        mapping_dict = {}
        for key in self.tags_dict:
            mapping_dict.update({key:key})
        print("creating the mapping dict ....")
        for key,value in tqdm(cluster.items()):
            new_value = max(value, key=value.get)
            for tag in value:
                mapping_dict.update({tag:new_value})
        print("updating the value...")
        return mapping_dict


    def get_classes_table(self):
        self.categories.replace(np.nan,"",inplace = True)
        self.categories.head()
        self.categories["variants"] = self.categories["Final Variants"]+","+self.categories["Similar Effects"]+","+self.categories["German"]
        self.categories["variants"] = self.categories["variants"].replace(np.nan,"")
    

    def get_near_concept_dict(self):
        """to get the keyword and near concept mapping """

        concept_dict = {}

        ##for effect:
        for keyword in self.categories["Effect"].dropna():
            concept_dict.update({keyword:"Effect"})

        ##for face Concerns
        for keyword,concept in zip(self.categories["Face Concerns"].dropna(),self.categories["Face Concerns(near concept)"].dropna()):
            concept_dict.update({keyword:concept})
        
        ##for Ingredients
        for keyword,concept in zip(self.categories["Ingredients\t"].dropna(),self.categories["Ingredients(near concept)"].dropna()):
            concept_dict.update({keyword:concept})


        concept_dict.update({"not in scope":"not defined"})
        return concept_dict


    def get_normalized_text(self,text):
        text_ = self.data_cleaning(text)
        ##preprocessing operations to normalize the sentenc
        words=text_.split()
        ## normalizing the sentence throw lemmatization and removing stopwords
        # words=[simplemma.lemmatize(word,lang=lang).strip() for word in words if word not in normalized_stopwords]
        words=[porter.stem(word.strip()) for word in words]# if word not in normalized_stopwords]
        # words=[lancaster.stem(word.strip()) for word in words]
        value = " ".join(words).strip()
        return value
  

    def get_keyword_dict(self):
        """for adding variants and defining the classes dict will all possible combination"""
        class_dict = {}
        ##for effects and variants
        for keyword,variants in zip(self.categories["Effect"],self.categories["variants"]):
            norm_keyword = self.get_normalized_text(keyword)
            if norm_keyword not in class_dict:
                class_dict.update({norm_keyword:[keyword]})
            else:
                if keyword not in class_dict[norm_keyword]:
                    class_dict[norm_keyword].append(keyword)

            if len(variants)>0:
                variant =[var for var in variants.split(",") if len(var)>0]
                for value in variant:
                    value = self.get_normalized_text(value)
                    if value not in class_dict:
                        class_dict.update({value :[keyword]})
                    else:
                        if keyword not in class_dict[value]:
                            class_dict[value].append(keyword)

        ##for face concerns and ingredients
        for key in ["Face Concerns","Ingredients\t"]:
            for keyword in self.categories[key]:
                norm_keyword = self.get_normalized_text(keyword)
                if norm_keyword not in class_dict:
                    class_dict.update({norm_keyword:[keyword]})
                else:
                    if keyword not in class_dict[norm_keyword]:
                        class_dict[norm_keyword].append(keyword)

        class_dict.pop("")
        return class_dict


    def get_class_flag(self,tag,class_):
        """flad to check whether the class in present in the tag or not"""
        flag = True
        spl_tag = {t:1 for t in tag.split()}
        spl_class = {c:1 for c in class_.split()}
        for word in spl_class:
            if word not in spl_tag:
                flag = False
        return flag


    def get_classes_from_tag_keyword_dict(self):
        tags_keyword_dict ={tag: [] for tag in self.tags_dict}
        for tag in tqdm(self.tags_dict):
            for class_ in self.class_dict:  
                if self.get_class_flag(self.tags_dict[tag],class_):
                    tags_keyword_dict[tag]+= self.class_dict[class_]
            if tags_keyword_dict[tag]==[]:
                tags_keyword_dict[tag].append("not in scope")

        return tags_keyword_dict

    def main(self):
        self.df["tags"] = self.df["text"].progress_apply(lambda review: self.get_tags_for_reviews(review))
        self.df.dropna(how="any",axis=0,inplace=True)
        self.df["tags"] = self.df["tags"].progress_apply(lambda tag:self.clean_tags_remove_unigrams(tag))
        self.df =self.df.explode(["tags"], ignore_index=True)
        self.tags_dict = dict(self.df["tags"].value_counts())
        cluster = self.Normalize()

        """for only selecting the clusters having more than one keyword
        and also getting the tag which have highest frequency"""
        cluster = {key: dict(sorted(value.items(), key=lambda item: item[1],reverse=True)) for key,value in cluster.items() if len(value)>1}
        print("length of output data",len(cluster)) 


        tag_mapping_dict = self.get_mapping_dict(cluster)
        self.df["tags"] = self.df["tags"].progress_apply(lambda tag: tag_mapping_dict[tag])

        self.tags_dict = dict(self.df["tags"].value_counts())
        self.tags_dict = {tag:self.get_normalized_text(tag) for tag in self.tags_dict}
        tags_keyword_dict = self.get_classes_from_tag_keyword_dict()
        self.df["class"] = self.df["tags"].progress_apply(lambda tag: tags_keyword_dict[tag])
        self.df=self.df.explode(["class"], ignore_index=True)

        return self.df




