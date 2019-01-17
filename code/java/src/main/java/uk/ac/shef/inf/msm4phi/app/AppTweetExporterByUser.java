package uk.ac.shef.inf.msm4phi.app;

import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.core.CoreContainer;
import uk.ac.shef.inf.msm4phi.IndexAnalyserMaster;
import uk.ac.shef.inf.msm4phi.Util;
import uk.ac.shef.inf.msm4phi.analysis.UserFeatureExporter;
import uk.ac.shef.inf.msm4phi.export.WorkerTweetExportByDays;
import uk.ac.shef.inf.msm4phi.export.WorkerTweetExportByUserType;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

public class AppTweetExporterByUser {
    public static void main(String[] args) throws IOException, SolrServerException {
        CoreContainer solrContainer = new CoreContainer(args[0]);
        solrContainer.load();

        SolrClient tweetIndex = new EmbeddedSolrServer(solrContainer.getCore("tweets"));
        SolrClient userIndex = new EmbeddedSolrServer(solrContainer.getCore("users"));

        WorkerTweetExportByUserType worker =
                new WorkerTweetExportByUserType(0, userIndex, tweetIndex,
                        args[1], 0.5,
                        "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/nodexl");
        IndexAnalyserMaster exporter = new IndexAnalyserMaster(
                new File("/home/zz/Work/msm4phi/data/symplur_hashtags/2_processed_hashtags.tsv"),
                worker
        );
        exporter.setThreads(1);
        exporter.process();
        tweetIndex.close();
        userIndex.close();
        System.exit(0);
    }

}
