package uk.ac.shef.inf.msm4phi.app;

import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrServerException;
import uk.ac.shef.inf.msm4phi.Util;
import uk.ac.shef.inf.msm4phi.indexing.AnnotationAppender;

import java.io.IOException;
import java.nio.file.Paths;

public class AppAnnotationAppender {

    public static void main(String[] args) throws IOException, SolrServerException {
        SolrClient solrClient =
                Util.getSolrClient(Paths.get(args[0]), "users");
        AnnotationAppender aa = new AnnotationAppender();
        aa.process(args[1],Integer.valueOf(args[2]),Integer.valueOf(args[3]),solrClient);
        System.exit(0);
    }
}
