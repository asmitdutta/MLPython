#!/usr/bin/python

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

#from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
#, SGDClassifier
#from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from scipy.stats import mode

from nltk.tokenize import word_tokenize
import urllib2

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

#short_pos = urllib2.urlopen('https://pythonprogramming.net/static/downloads/short_reviews/positive.txt').read()
#short_neg = open("short_reviews/negative.txt","r").read()
#short_neg = urllib2.urlopen('https://pythonprogramming.net/static/downloads/short_reviews/negative.txt').read()
short_neg = """
simplistic , silly and tedious .
it's so laddish and juvenile , only teenage boys could possibly find it funny .
exploitative and largely devoid of the depth or sophistication that would make watching such a graphic treatment of the crimes bearable .
[garbus] discards the potential for pathological study , exhuming instead , the skewed melodrama of the circumstantial situation .
a visually flashy but narratively opaque and emotionally vapid exercise in style and mystification .
the story is also as unoriginal as they come , already having been recycled more times than i'd care to count .
about the only thing to give the movie points for is bravado -- to take an entirely stale concept and push it through the audience's meat grinder one more time .
not so much farcical as sour .
unfortunately the story and the actors are served with a hack script .
all the more disquieting for its relatively gore-free allusions to the serial murders , but it falls down in its attempts to humanize its subject .
a sentimental mess that never rings true .
while the performances are often engaging , this loose collection of largely improvised numbers would probably have worked better as a one-hour tv documentary .
interesting , but not compelling .
on a cutting room floor somewhere lies . . . footage that might have made no such thing a trenchant , ironic cultural satire instead of a frustrating misfire .
while the ensemble player who gained notice in guy ritchie's lock , stock and two smoking barrels and snatch has the bod , he's unlikely to become a household name on the basis of his first starring vehicle .
there is a difference between movies with the courage to go over the top and movies that don't care about being stupid
nothing here seems as funny as it did in analyze this , not even joe viterelli as de niro's right-hand goombah .
such master screenwriting comes courtesy of john pogue , the yale grad who previously gave us " the skulls " and last year's " rollerball . " enough said , except : film overboard !
here , common sense flies out the window , along with the hail of bullets , none of which ever seem to hit sascha .
this 100-minute movie only has about 25 minutes of decent material .
the execution is so pedestrian that the most positive comment we can make is that rob schneider actually turns in a pretty convincing performance as a prissy teenage girl .
on its own , it's not very interesting . as a remake , it's a pale imitation .
it shows that some studios firmly believe that people have lost the ability to think and will forgive any shoddy product as long as there's a little girl-on-girl action .
a farce of a parody of a comedy of a premise , it isn't a comparison to reality so much as it is a commentary about our knowledge of films .
as exciting as all this exoticism might sound to the typical pax viewer , the rest of us will be lulled into a coma .
the party scenes deliver some tawdry kicks . the rest of the film . . . is dudsville .
"""
short_pos = """the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .
the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .
effective but too-tepid biopic
if you sometimes like to go to the movies to have fun , wasabi is a good place to start .
emerges as something rare , an issue movie that's so honest and keenly observed that it doesn't feel like one .
the film provides some great insight into the neurotic mindset of all comics -- even those who have reached the absolute top of the game .
offers that rare combination of entertainment and education .
perhaps no picture ever made has more literally showed that the road to hell is paved with good intentions .
steers turns in a snappy screenplay that curls at the edges ; it's so clever you want to hate it . but he somehow pulls it off .
take care of my cat offers a refreshingly different slice of asian cinema .
this is a film well worth seeing , talking and singing heads and all .
what really surprises about wisegirls is its low-key quality and genuine tenderness .
 ( wendigo is ) why we go to the cinema : to be fed through the eye , the heart , the mind .
one of the greatest family-oriented , fantasy-adventure movies ever .
ultimately , it ponders the reasons we need stories so much .
an utterly compelling 'who wrote it' in which the reputation of the most famous author who ever lived comes into question .
illuminating if overly talky documentary .
a masterpiece four years in the making .
the movie's ripe , enrapturing beauty will tempt those willing to probe its inscrutable mysteries .
offers a breath of the fresh air of true sophistication .
a thoughtful , provocative , insistently humanizing film .
with a cast that includes some of the top actors working in independent film , lovely & amazing involves us because it is so incisive , so bleakly amusing about how we go about our lives .
a disturbing and frighteningly evocative assembly of imagery and hypnotic music composed by philip glass .
not for everyone , but for those with whom it will connect , it's a nice departure from standard moviegoing fare .
scores a few points for doing what it does with a dedicated and good-hearted professionalism .
occasionally melodramatic , it's also extremely effective .
spiderman rocks
an idealistic love story that brings out the latent 15-year-old romantic in everyone .
at about 95 minutes , treasure planet maintains a brisk pace as it races through the familiar story . however , it lacks grandeur and that epic quality often associated with stevenson's tale as well as with earlier disney efforts .
it helps that lil bow wow . . . tones down his pint-sized gangsta act to play someone who resembles a real kid .
guaranteed to move anyone who ever shook , rattled , or rolled .
a masterful film from a master filmmaker , unique in its deceptive grimness , compelling in its fatalist worldview .
light , cute and forgettable .
if there's a way to effectively teach kids about the dangers of drugs , i think it's in projects like the ( unfortunately r-rated ) paid .
while it would be easy to give crush the new title of two weddings and a funeral , it's a far more thoughtful film than any slice of hugh grant whimsy .
though everything might be literate and smart , it never took off and always seemed static .
cantet perfectly captures the hotel lobbies , two-lane highways , and roadside cafes that permeate vincent's days
ms . fulford-wierzbicki is almost spooky in her sulky , calculating lolita turn .
though it is by no means his best work , laissez-passer is a distinguished and distinctive effort by a bona-fide master , a fascinating film replete with rewards to be had by all willing to make the effort to reap them .
like most bond outings in recent years , some of the stunts are so outlandish that they border on being cartoonlike . a heavy reliance on cgi technology is beginning to creep into the series .
newton draws our attention like a magnet , and acts circles around her better known co-star , mark wahlberg .
the story loses its bite in a last-minute happy ending that's even less plausible than the rest of the picture . much of the way , though , this is a refreshingly novel ride .
fuller would surely have called this gutsy and at times exhilarating movie a great yarn .
the film makes a strong case for the importance of the musicians in creating the motown sound .
karmen moves like rhythm itself , her lips chanting to the beat , her long , braided hair doing little to wipe away the jeweled beads of sweat .
gosling provides an amazing performance that dwarfs everything else in the film .
a real movie , about real people , that gives us a rare glimpse into a culture most of us don't know .
tender yet lacerating and darkly funny fable .
may be spoofing an easy target -- those old '50's giant creature features -- but . . . it acknowledges and celebrates their cheesiness as the reason why people get a kick out of watching them today .
an engaging overview of johnson's eccentric career .
in its ragged , cheap and unassuming way , the movie works .
some actors have so much charisma that you'd be happy to listen to them reading the phone book . hugh grant and sandra bullock are two such likeable actors .
sandra nettelbeck beautifully orchestrates the transformation of the chilly , neurotic , and self-absorbed martha as her heart begins to open .
behind the snow games and lovable siberian huskies ( plus one sheep dog ) , the picture hosts a parka-wrapped dose of heart .
everytime you think undercover brother has run out of steam , it finds a new way to surprise and amuse .
manages to be original , even though it rips off many of its ideas .
you'd think by now america would have had enough of plucky british eccentrics with hearts of gold . yet the act is still charming here .
whether or not you're enlightened by any of derrida's lectures on " the other " and " the self , " derrida is an undeniably fascinating and playful fellow .
a pleasant enough movie , held together by skilled ensemble actors .
this is the best american movie about troubled teens since 1998's whatever .
disney has always been hit-or-miss when bringing beloved kids' books to the screen . . . tuck everlasting is a little of both .
just the labour involved in creating the layered richness of the imagery in this chiaroscuro of madness and light is astonishing .
the animated subplot keenly depicts the inner struggles of our adolescent heroes - insecure , uncontrolled , and intense .
the invincible werner herzog is alive and well and living in la
morton is a great actress portraying a complex character , but morvern callar grows less compelling the farther it meanders from its shocking start .
part of the charm of satin rouge is that it avoids the obvious with humour and lightness .
son of the bride may be a good half-hour too long but comes replete with a flattering sense of mystery and quietness .
a simmering psychological drama in which the bursts of sudden violence are all the more startling for the slow buildup that has preceded them .
a taut , intelligent psychological drama .
a compelling coming-of-age drama about the arduous journey of a sensitive young girl through a series of foster homes and a fierce struggle to pull free from her dangerous and domineering mother's hold over her .
a truly moving experience , and a perfect example of how art -- when done right -- can help heal , clarify , and comfort .
this delicately observed story , deeply felt and masterfully stylized , is a triumph for its maverick director .
at heart the movie is a deftly wrought suspense yarn whose richer shadings work as coloring rather than substance .
the appearance of treebeard and gollum's expanded role will either have you loving what you're seeing , or rolling your eyes . i loved it ! gollum's 'performance' is incredible !
a screenplay more ingeniously constructed than " memento "
if this movie were a book , it would be a page-turner , you can't wait to see what happens next .
haneke challenges us to confront the reality of sexual aberration .
absorbing and disturbing -- perhaps more disturbing than originally intended -- but a little clarity would have gone a long way .
it's the best film of the year so far , the benchmark against which all other best picture contenders should be measured .
painful to watch , but viewers willing to take a chance will be rewarded with two of the year's most accomplished and riveting film performances .
this is a startling film that gives you a fascinating , albeit depressing view of iranian rural life close to the iraqi border .
an imaginative comedy/thriller .
a few artsy flourishes aside , narc is as gritty as a movie gets these days .
while the isle is both preposterous and thoroughly misogynistic , its vistas are incredibly beautiful to look at .
together , tok and o orchestrate a buoyant , darkly funny dance of death . in the process , they demonstrate that there's still a lot of life in hong kong cinema .
director kapur is a filmmaker with a real flair for epic landscapes and adventure , and this is a better film than his earlier english-language movie , the overpraised elizabeth .
the movie is a blast of educational energy , as bouncy animation and catchy songs escort you through the entire 85 minutes .
a sports movie with action that's exciting on the field and a story you care about off it .
doug liman , the director of bourne , directs the traffic well , gets a nice wintry look from his locations , absorbs us with the movie's spycraft and uses damon's ability to be focused and sincere .
the tenderness of the piece is still intact .
katz uses archival footage , horrifying documents of lynchings , still photographs and charming old reel-to-reel recordings of meeropol entertaining his children to create his song history , but most powerful of all is the song itself
like the film's almost anthropologically detailed realization of early-'80s suburbia , it's significant without being overstated . """

documents = []

for r in short_pos.split('\n'):
    documents.append( (r, "pos") )

for r in short_neg.split('\n'):
    documents.append( (r, "neg") )


all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = w in words
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

# positive data example:
training_set = featuresets[:10000]
testing_set =  featuresets[10000:]

##
### negative data example:
##training_set = featuresets[100:]
##testing_set =  featuresets[:100]

print "training_set",training_set
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)