package uk.ac.shef.inf.wop.indexing;

import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.core.CoreContainer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ForkJoinPool;

public class NTripleIndexerApp{
    private static final Logger LOG = Logger.getLogger(NTripleIndexerApp.class.getName());

    public static void main(String[] args) throws IOException {

        CoreContainer solrContainer = new CoreContainer(args[1]);
        solrContainer.load();

        SolrClient entitiesCoreClient = new EmbeddedSolrServer(solrContainer.getCore("entities"));
        //SolrClient predicatesCoreClient= new EmbeddedSolrServer(solrContainer.getCore("predicates"));
        List<String> gzFiles = new ArrayList<>();
        for (File f: new File(args[0]).listFiles())
            gzFiles.add(f.toString());
        LOG.info("Initialisation completed.");
        NTripleIndexerWorker worker = new NTripleIndexerWorker(0,entitiesCoreClient, null,gzFiles,
                Long.valueOf(args[2]), Long.valueOf(args[3]), Boolean.valueOf(args[4]));

        try {

            ForkJoinPool forkJoinPool = new ForkJoinPool();
            int total = forkJoinPool.invoke(worker);

            LOG.info(String.format("Completed, total entities=%s", total, new Date().toString()));

        } catch (Exception ioe) {
            StringBuilder sb = new StringBuilder("Failed to build features!");
            sb.append("\n").append(ExceptionUtils.getFullStackTrace(ioe));
            LOG.error(sb.toString());
        }


        entitiesCoreClient.close();
        System.exit(0);
    }

}