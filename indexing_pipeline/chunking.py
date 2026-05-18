from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import NodeRelationship
from indexing_pipeline.schema import LectureDocument

class HierarchicalChunker:
    def __init__(self):
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes = [1024, 256],
            chunk_overlap = 20
        )
       

    def _parse(self, document):
        all_nodes = self.node_parser.get_nodes_from_documents([document])
        child_nodes = get_leaf_nodes(all_nodes)
        
        # Identify Parents (nodes that are not leaves)
        parent_nodes = [n for n in all_nodes if n not in child_nodes]
        # print("child nodes")
        # print(child_nodes)
        # print("parent nodes")
        # print(parent_nodes)
        return child_nodes, parent_nodes
    
    def _create_document(self, document: LectureDocument):

        metadata = document.model_dump(
            exclude={"transcript"}
        )
        created_document = Document(
            text = document.transcript,
            metadata = metadata
        )
        return created_document
    

    def process_document(self, document: LectureDocument):
        created_document = self._create_document(document)
        child_nodes, parent_nodes = self._parse(created_document)
        return child_nodes, parent_nodes
        
if __name__ == "__main__":

    chunker = HierarchicalChunker()
    raw_data = {'id':"59fb40f7e3556cf2a61aa632d2418721",'url': 'https://www.thespiritualscientist.com/how-can-humility-go-along-with-self-respect/', 'title': 'Humily and self respect', 'transcript': 'Answer:Let’s look at it this way.  Is there a difference between humility and humiliation?  Yes, there is, definitely.\n\nWe say we should cultivate humility, but none of us want to be humiliated.  Let’s put it another way, we all want to cultivate humility.  So, should we start insulting and humiliating each other in our community?  Then we will all become humble.  You are proud and I will insult you, I will humiliate you and you humiliate me and in this way by exchanging humiliation we will all get humility.  No!\n\nDoes humiliation make one humble?  Not necessarily.  Humiliation can make one feel offended, it can make one feel enraged.  Sometimes it may make one humble, but not necessarily.  So, clearly, there is a difference between humiliation and humility, and certainly although we talk about, say, we should be humble, but we also say respect each other. That’s the injunction in our devotee community and that’s normal human conduct also.  So, we could put it that humiliation is false ego frustrated.  Humility is false ego rejected.\n\nHumiliation is false ego frustrated.  I want to be respected, but instead of being respected, I was disrespected, I was mocked, I was derided, I can’t bear this.  So, humiliation is false ego frustrated.  I want to be respected, but I was not.\n\nBut humility is false ego rejected.  That means I don’t crave for respect from others.  I don’t depend on respect from others.  That doesn’t mean that I don’t care at all.  I mean it’s not so easily possible.  We are human beings and we will notice how people are dealing with us.  We can’t artificially become stone-like and that’s not exactly humility.  But humility and humiliation are… Humiliation is I want something and I don’t get that respect, that’s humiliation.  But humility is I don’t want it that much.\n\nSo, we could look at humility from the perspective of what we expect, what we demand from the world, what we need from the world.  So that’s one aspect of humility.\n\nAnother aspect is, if I’m not demanding, how am I looking at myself?  So now we have great saintly people saying that, Bhaktivinoda Thakura says “Amara Jeevana Sada Pape Rata,” that my life is full of sin and there is no good that is seen in me.  Now, he has songs like that, but then he is writing books, he is sharing Bhakti wisdom confidently, countering misconceptions.  So, when he is saying there are no good qualities in me, I am sinful, he is looking at it from a very elevated perspective.  So, he is thinking of how pure Krishna’s devotees are, how pure Krishna is.  As compared to them, what am I?  Krishna loves me so much, what am I doing to reciprocate with him?  I am doing nothing.\n\nSo, when we are at a different level of consciousness, we are not really perceiving how much Krishna loves us.  Often, we may actually be feeling the opposite.  Why doesn’t Krishna care for me?  There are so many things wrong in my life, why is Krishna not helping me? Does Krishna even care for me?  So, when we are not at that level, we can’t artificially imitate that.\n\nOur frame of reference presently is mostly our human society and how people are interacting with each other and people are interacting with us in human society.  But for great souls like Bhaktivinoda Thakura or Krishnadas Kaviraj Goswami, their frame of reference is not human society.  Their frame of reference is Krishna and community of exalted devotees of Krishna.  As compared to that, they feel, what about me, I am nobody. But in human society, they are functioning assertively, they are functioning very purposefully and even strongly when necessary for Krishna’s service.  So, how do we look at ourselves?  One aspect of humility is that it’s not that we look down at ourselves, we think I am worthless, I am useless.  Well, we see that I am a part of Krishna and in that sense there is intrinsic worth for me.  I have a whole seminar on “can we love ourselves”.  Self-love can seem very self-indulgent, but actually it’s not.\n\nIf we understand we are parts of Krishna, how can we love Krishna without loving his parts? The part over which we have most control, the part for which we are most responsible is we ourselves.  Now we don’t love Krishna in the sense of becoming self-indulgent and I am great.  No, but love in the sense of that we respect, we care for ourselves, we respect ourselves, we want the best for ourselves.  So, that is definitely there.  Humility doesn’t mean we look down at ourselves, rather we are not looking at ourselves constantly and not looking at how people are looking at us.  We are thinking of Krishna and we are thinking of how we can serve Krishna.\n\nWe are thinking of what is my service, what is my responsibility, how can I do it the best.  In that sense, self-respect and humility go together because we don’t need other people’s respect because we have that intrinsic self-respect.', 'soundcloud_links': ['https://soundcloud.com/chaitanya-charan/how-can-humility-go-along-with-self-respect'], 'youtube_links': []}

    doc = LectureDocument(**raw_data)

    chunker.process_document(doc)
