package uk.ac.shef.inf.msm4phi.app;

import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.core.CoreContainer;
import uk.ac.shef.inf.msm4phi.indexing.IndexPopulatorUser;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.IOException;
import java.nio.file.Paths;

/**
 * This should be used only after 'favorite' and 'retweet' stats are collected as a postprocess, see AppPostProcess
 */
public class AppUserIndexGenerator {
    public static void main(String[] args) throws IOException {
        CoreContainer solrContainer = new CoreContainer(args[0]);
        solrContainer.load();

        SolrClient tweetSolrClient = new EmbeddedSolrServer(solrContainer.getCore("tweets"));
        SolrClient userSolrClient= new EmbeddedSolrServer(solrContainer.getCore("users"));

        IndexPopulatorUser pprocess= new IndexPopulatorUser(tweetSolrClient,userSolrClient,
                args[1],args[2],args[3],args[4]);
        pprocess.process();
        tweetSolrClient.close();
        userSolrClient.close();

        System.exit(0);
    }
}
