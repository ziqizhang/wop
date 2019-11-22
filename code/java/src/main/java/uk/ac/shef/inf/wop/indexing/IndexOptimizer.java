package uk.ac.shef.inf.wop.indexing;

import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.core.CoreContainer;

import java.io.IOException;
import java.util.Date;

public class IndexOptimizer {

    public static void main(String[] args) throws IOException, SolrServerException {
        CoreContainer solrContainer = new CoreContainer(args[0]);
        solrContainer.load();

        SolrClient entitiesCoreClient = new EmbeddedSolrServer(solrContainer.getCore("entities"));

        System.out.println(String.format("Optimising the index ... %s", new Date().toString()));
        entitiesCoreClient.optimize();
        entitiesCoreClient.close();
        System.exit(0);
    }
}
