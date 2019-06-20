package uk.ac.shef.inf.wop.exporting;

import org.apache.commons.lang.StringEscapeUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.params.CursorMarkParams;
import org.apache.solr.core.CoreContainer;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * This class reads the index created by ProdDescExporter, to export product descriptions into batches of txt files
 */
public class ProdDescTextFileExporter_Cursor implements Runnable {

    private static final Logger LOG = Logger.getLogger(ProdDescTextFileExporter.class.getName());
    private long maxWordsPerFile = 5000000000L;
    //private long maxWordsPerFile=500;
    private int nameFileCounter = 0, descFileCounter = 0;
    private PrintWriter nameFile;
    private PrintWriter descFile;

    private int id;
    private int start;
    private int end;
    private SolrClient prodNameDescIndex;
    private int resultBatchSize;
    private String nameOutFolder;
    private String descOutFolder;

    public ProdDescTextFileExporter_Cursor(int id, int start, int end,
                                    SolrClient prodNameDescIndex,
                                    int resultBatchSize, String nameOutFolder,
                                    String descOutFolder) {
        this.id=id;
        this.start=start;
        this.end=end;
        this.prodNameDescIndex=prodNameDescIndex;
        this.resultBatchSize=resultBatchSize;
        this.nameOutFolder=nameOutFolder;
        this.descOutFolder=descOutFolder;
    }

    public void run()  {
        SolrQuery q = createQuery(resultBatchSize, start);
        String cursorMark = CursorMarkParams.CURSOR_MARK_START;
        QueryResponse res=null;
        boolean stop = false;
        long total = 0;

        LOG.info(String.format("\tthread %d: Started, begin=%d end=%d...",
                id, q.getStart(), end));

        try {
            nameFile = new PrintWriter(new FileWriter(nameOutFolder + "/n_" + id + "_" + nameFileCounter, true));
            descFile = new PrintWriter(new FileWriter(descOutFolder + "/d_" + id + "_" + descFileCounter, true));

            long countNameFileWords = 0, countDescFileWords = 0, curr = 0;

            while (!stop) {
                try {
                    q.set(CursorMarkParams.CURSOR_MARK_PARAM, cursorMark);
                    res = prodNameDescIndex.query(q);

                    if (res != null)
                        total = res.getResults().getNumFound();
                    //update results
                    LOG.info(String.format("\t\tthread %d: total results of %d, currently processing from %d to %d /started at %d...",
                            id, total, q.getStart(), q.getStart() + q.getRows(), start));

                    for (SolrDocument d : res.getResults()) {
                        //process and export to the other solr index
                        long[] words = exportRecord(d, nameFile, descFile);
                        countNameFileWords += words[0];
                        countDescFileWords += words[1];
                    }

                    if (countNameFileWords >= maxWordsPerFile) {
                        nameFile.close();
                        nameFileCounter++;
                        nameFile = new PrintWriter(new FileWriter(nameOutFolder + "/n_" + nameFileCounter, true));
                        countNameFileWords = 0;
                    }
                    if (countDescFileWords >= maxWordsPerFile) {
                        descFile.close();
                        descFileCounter++;
                        descFile = new PrintWriter(new FileWriter(descOutFolder + "/d_" + descFileCounter, true));
                        countDescFileWords = 0;
                    }

                } catch (Exception e) {
                    LOG.warn(String.format("\t\t thread %d unable to successfully index product triples starting from index %s. Due to error: %s",
                            id, start,
                            ExceptionUtils.getFullStackTrace(e)));

                }

                curr += q.getRows();
                if (curr < end) {
                    String nextCursorMark = res.getNextCursorMark();
                    if (cursorMark.equals(nextCursorMark)) {
                        break;
                    }
                    cursorMark = nextCursorMark;
                }
                else {
                    stop = true;
                    LOG.info("\t\t thread "+id+" reached the end. Stopping...");
                }
            }

            try {
                nameFile.close();
                descFile.close();
            } catch (Exception e) {
                LOG.warn(String.format("\t\t thread %d unable to shut down servers due to error: %s",
                        id, ExceptionUtils.getFullStackTrace(e)));
            }
        }catch (IOException ioe){
            LOG.warn(String.format("\t\t thread %d unable to create output files, io exception: %s",
                    id, ExceptionUtils.getFullStackTrace(ioe)));
        }
    }

    private long[] exportRecord(SolrDocument d,
                                PrintWriter nameFile, PrintWriter descFile) {

        Object nameData = d.getFieldValue("name");
        Object descData = d.getFirstValue("text");
        long[] res = new long[2];

        if (nameData != null) {
            String name = cleanData(nameData.toString());
            long tokens = name.split("\\s+").length;
            if (name.length() > 10 && tokens > 2) {
                nameFile.println(name);
                res[0] = tokens;
            }
        }
        if (descData != null) {
            String desc = cleanData(descData.toString());
            long tokens = desc.split("\\s+").length;
            if (desc.length() > 20 && tokens > 5) {
                descFile.println(desc);
                res[1] = tokens;
            }
        }

        return res;
    }

    private String cleanData(String value) {
        value = StringEscapeUtils.unescapeJava(value);
        value = value.replaceAll("\\s+", " ");
        value = StringUtils.stripAccents(value);
        return value.trim();
    }

    private static SolrQuery createQuery(int resultBatchSize, int start) {
        SolrQuery query = new SolrQuery();
        query.setQuery("text:* OR name:*");
        //query.setSort("id", SolrQuery.ORDER.asc);
        //query.setStart(start);
        query.setRows(resultBatchSize);

        return query;
    }

    public static void main(String[] args) throws IOException {
        CoreContainer prodNDContainer = new CoreContainer(args[0]);
        prodNDContainer.load();
        SolrClient prodNameDescIndex = new EmbeddedSolrServer(prodNDContainer.getCore("proddesc"));

        //74488335
        int jobStart=Integer.valueOf(args[4]);
        int jobs = Integer.valueOf(args[5]);

        int threads=Integer.valueOf(args[3]);
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        for (int i = 0; i < threads; i++) {
            int start=jobStart+i*jobs;
            int end=start+jobs;
            Runnable exporter = new ProdDescTextFileExporter_Cursor(i,
                    start,end,
                    prodNameDescIndex,
                    5000,
                    args[1],args[2]);
            executor.execute(exporter);
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        prodNameDescIndex.close();
        System.exit(0);
        LOG.info("COMPLETE!");

    }
}
