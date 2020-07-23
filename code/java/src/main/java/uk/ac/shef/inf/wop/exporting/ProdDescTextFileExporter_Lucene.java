package uk.ac.shef.inf.wop.exporting;

import org.apache.commons.lang.StringEscapeUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Logger;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.solr.core.CoreContainer;
import org.apache.solr.search.SolrIndexSearcher;
import uk.ac.shef.inf.wop.Util;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * This class reads the index created by ProdDescExporter, to export product descriptions into batches of txt files
 */
public class ProdDescTextFileExporter_Lucene implements Runnable {

    private static final Logger LOG = Logger.getLogger(ProdDescTextFileExporter.class.getName());
    private long maxWordsPerFile = 1000000000L;
    //private long maxWordsPerFile=500;
    private int nameFileCounter = 0, descFileCounter = 0;
    private PrintWriter nameFile;
    private PrintWriter descFile;

    private int id;
    private int start;
    private int end;
    private IndexReader luceneIndexReader;
    private int resultBatchSize;
    private String nameOutFolder;
    private String descOutFolder;

    private static final int MIN_DESC_WORDS=30;

    public ProdDescTextFileExporter_Lucene(int id, int start, int end,
                                           IndexReader prodNameDescIndex,
                                           int resultBatchSize, String nameOutFolder,
                                           String descOutFolder) {
        this.id = id;
        this.start = start;
        this.end = end;
        this.luceneIndexReader = prodNameDescIndex;
        this.resultBatchSize = resultBatchSize;
        this.nameOutFolder = nameOutFolder;
        this.descOutFolder = descOutFolder;
    }

    public void run() {
        long total = 0;

        LOG.info(String.format("\tthread %d: Started, begin=%d end=%d...",
                id, start, end));

        try {
            nameFile = new PrintWriter(new FileWriter(nameOutFolder + "/n_" + id + "_" + nameFileCounter, true));
            descFile = new PrintWriter(new FileWriter(descOutFolder + "/d_" + id + "_" + descFileCounter, true));

            long countNameFileWords = 0, countDescFileWords = 0, curr = 0;
            //update results

            for (int i = start; i < luceneIndexReader.maxDoc() && i < end; i++) {

                Document doc = luceneIndexReader.document(i);
                long[] words = exportRecord(doc, nameFile, descFile);
                countNameFileWords += words[0];
                countDescFileWords += words[1];
                if (i%resultBatchSize==0)
                    LOG.info(String.format("\t\tthread %d: total results of %d, currently processing %d /started at %d to %d...",
                        id, total, i, start,end));
                // do something with docId here...

                if (countNameFileWords >= maxWordsPerFile) {
                    LOG.info(String.format("\t\tthread %d: finishing name file, total words= %d",
                            id, countNameFileWords));
                    nameFile.close();
                    nameFileCounter++;
                    nameFile = new PrintWriter(new FileWriter(nameOutFolder + "/n_" + id + "_"+ nameFileCounter, true));
                    countNameFileWords = 0;
                }
                if (countDescFileWords >= maxWordsPerFile) {
                    LOG.info(String.format("\t\tthread %d: finishing desc file, total words= %d",
                            id, countDescFileWords));
                    descFile.close();
                    descFileCounter++;
                    descFile = new PrintWriter(new FileWriter(descOutFolder + "/d_" + id + "_"+ descFileCounter, true));
                    countDescFileWords = 0;
                }
            }


            try {
                nameFile.close();
                descFile.close();
                LOG.info(String.format("\t\tthread %d: finishing name file, total words= %d",
                        id, countNameFileWords));
                LOG.info(String.format("\t\tthread %d: finishing desc file, total words= %d",
                        id, countDescFileWords));
            } catch (Exception e) {
                LOG.warn(String.format("\t\t thread %d unable to shut down servers due to error: %s",
                        id, ExceptionUtils.getFullStackTrace(e)));
            }
        } catch (IOException ioe) {
            LOG.warn(String.format("\t\t thread %d unable to create output files, io exception: %s",
                    id, ExceptionUtils.getFullStackTrace(ioe)));
        }
    }

    private long[] exportRecord(Document d,
                                PrintWriter nameFile, PrintWriter descFile) {

        String nameData = d.get("name");
        String descData = d.get("desc");
        long[] res = new long[2];

        if (nameData != null) {
            String name = cleanData(nameData);
            long tokens = name.split("\\s+").length;
            if (name.length() > 15 && tokens > 3) {
                nameFile.println(name);
                res[0] = tokens;
            }
        }
        if (descData != null) {
            String desc = cleanData(descData);
            long tokens = desc.split("\\s+").length;
            if (desc.length() > 20 && tokens >= MIN_DESC_WORDS) {
                descFile.println(desc);
                res[1] = tokens;
            }
        }

        return res;
    }

    /**
     * @param value
     * @return
     */
    private String cleanData(String value) {
        if (value.startsWith("<")&&value.endsWith(">"))
            return "";

        try {
            value = StringEscapeUtils.unescapeJava(value);
        }catch (Exception e){}
        //removes all non-alphanumeric-or-punctuation characters
        value = value.replaceAll("[^\\p{IsAlphabetic}\\p{IsDigit}\\p{Punct}]", " ").
                replaceAll("\\s+", " ").trim();
        value = StringUtils.stripAccents(value);

        return value.trim();
    }

    public static void main(String[] args) throws IOException {
        CoreContainer prodNDContainer = new CoreContainer(args[0]);
        prodNDContainer.load();
        SolrIndexSearcher solrIndexSearcher= prodNDContainer.getCore("prodcatdesc").getSearcher().get();
        IndexReader prodNameDescIndex = solrIndexSearcher.getIndexReader();
        //74488335
        int jobStart = Integer.valueOf(args[4]);
        int jobs = Integer.valueOf(args[5]);

        int threads = Integer.valueOf(args[3]);
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        for (int i = 0; i < threads; i++) {
            int start = jobStart + i * jobs;
            int end = start + jobs;
            Runnable exporter = new ProdDescTextFileExporter_Lucene(i,
                    start, end,
                    prodNameDescIndex,
                    5000,
                    args[1], args[2]);
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
