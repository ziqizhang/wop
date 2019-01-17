package uk.ac.shef.inf.msm4phi.app;

import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrServerException;
import uk.ac.shef.inf.msm4phi.Util;
import uk.ac.shef.inf.msm4phi.analysis.UserFeatureExporter;

import java.io.IOException;
import java.nio.file.Paths;

public class AppUserFeatureExport{
    public static void main(String[] args) throws IOException, SolrServerException {
        SolrClient solrClient =
                Util.getSolrClient(Paths.get(args[0]), "users");
        UserFeatureExporter exporter = new UserFeatureExporter();
        exporter.process(Integer.MAX_VALUE, args[1], solrClient);
        solrClient.close();
    }

}
