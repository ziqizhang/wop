package uk.ac.shef.inf.msm4phi.app;

import org.apache.solr.client.solrj.SolrClient;
import uk.ac.shef.inf.msm4phi.IndexAnalyserMaster;
import uk.ac.shef.inf.msm4phi.IndexAnalyserWorker;
import uk.ac.shef.inf.msm4phi.Util;
import uk.ac.shef.inf.msm4phi.stats.content.ContentStatsExtractor;
import uk.ac.shef.inf.msm4phi.stats.interaction.InteractionStatsExtractor;
import uk.ac.shef.inf.msm4phi.stats.user.UserStatsExtractor;
import uk.ac.shef.inf.msm4phi.stats.user.UserTypeStatsExtractor;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;


public class AppStatsExtractor {
    public static void main(String[] args) throws IOException {
        SolrClient solrClient=null;
        IndexAnalyserWorker worker=null;
        if (args[0].equalsIgnoreCase("c")){
            solrClient= Util.getSolrClient(Paths.get(args[1]),"tweets");
            worker = new ContentStatsExtractor(0,solrClient, args[3]);
        }else if (args[0].equalsIgnoreCase("i")){
            solrClient= Util.getSolrClient(Paths.get(args[1]),"tweets");
            worker = new InteractionStatsExtractor(0,solrClient, args[3]);
        }else if (args[0].equalsIgnoreCase("u")){
            solrClient= Util.getSolrClient(Paths.get(args[1]),"users");
            worker = new UserStatsExtractor(0,solrClient, args[3]);
        }else if (args[0].equalsIgnoreCase("ul")){
            solrClient= Util.getSolrClient(Paths.get(args[1]),"users");
            worker = new UserTypeStatsExtractor(0,solrClient, args[3]);
        }else{
            System.err.println("Not supported. Use 'c/i/u/ul'.");
            System.exit(1);
        }

        IndexAnalyserMaster master = new IndexAnalyserMaster(new File(args[2]), worker);
        master.process();

        solrClient.close();
    }
}
