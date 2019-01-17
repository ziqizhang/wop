package uk.ac.shef.inf.msm4phi.analysis;

import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.core.CoreContainer;

import java.io.IOException;

public class AnalysisMain {
    public static void main(String[] args) throws IOException {
        CoreContainer solrContainer = new CoreContainer(args[0]);
        solrContainer.load();

        SolrClient tweetSolrClient = new EmbeddedSolrServer(solrContainer.getCore("tweets"));
        SolrClient userSolrClient= new EmbeddedSolrServer(solrContainer.getCore("users"));

        //distribution stats
        //generateDistributionStats(tweetSolrClient, userSolrClient, args[1],args[2]);

        //create graph similarities
        //createGraphSimilarityMatrix(tweetSolrClient, userSolrClient, args[1], args[2]);

        //create tag coocurring stats
        generateFrequentCoocurringTags(tweetSolrClient, args[1],args[2]);

        tweetSolrClient.close();
        userSolrClient.close();

        System.exit(0);
    }

    private static void generateFrequentCoocurringTags(SolrClient tweetCore,
                                                       String hashtagFile, String outFolder) throws IOException {
        FrequentCooccurTagFinder finder = new FrequentCooccurTagFinder(null);
        finder.process(tweetCore,hashtagFile,outFolder+"/tag_coocurr.csv");
    }

    private static void generateDistributionStats(SolrClient tweetCore, SolrClient userCore,
                                                  String hashtagFile, String outFolder) throws IOException {
        HashtagCommunityPresence hcp = new HashtagCommunityPresence();
        hcp.process(hashtagFile, outFolder+"/presence-hashtag.csv", tweetCore);

        UserCommunityPresence ucp = new UserCommunityPresence();
        ucp.process(hashtagFile, outFolder+"/presence-user.csv", userCore);

        HashtagUserAwareness hua = new HashtagUserAwareness();
        hua.process(outFolder+"/tag-user-awareness.csv",userCore);
    }

    private static void createGraphSimilarityMatrix(SolrClient tweetCore, SolrClient userCore,
                                                    String hashtagFile, String outFolder) throws IOException {
        GraphDiseaseCoocurByTag gdt =
                new GraphDiseaseCoocurByTag(outFolder+"/distribution/presence-hashtag.csv");
        gdt.process(tweetCore,hashtagFile,outFolder+"/graphsim_tag.csv");

        GraphDiseaseCoocurByUser gdu =
                new GraphDiseaseCoocurByUser(outFolder+"/distribution/presence-user.csv");
        gdu.process(userCore, hashtagFile, outFolder+"/graphsim_user.csv");
        /*GraphDiseaseCoocurBySelectedTag gdst =
                new GraphDiseaseCoocurBySelectedTag(null);
        gdst.process(tweetCore,hashtagFile,outFolder+"/graphsim_stag.csv");*/

    }
}
