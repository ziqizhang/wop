package uk.ac.shef.inf.msm4phi.app;

import org.apache.solr.client.solrj.SolrClient;
import uk.ac.shef.inf.msm4phi.indexing.PostProcessor;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.IOException;
import java.nio.file.Paths;

public class AppPostProcess {
    public static void main(String[] args) throws IOException {
        SolrClient solrClient= Util.getSolrClient(Paths.get(args[0]),"tweets");
        PostProcessor pprocess= new PostProcessor(solrClient,
                args[1],args[2],args[3],args[4]);
        pprocess.process();

        solrClient.close();
        System.exit(0);
    }
}
