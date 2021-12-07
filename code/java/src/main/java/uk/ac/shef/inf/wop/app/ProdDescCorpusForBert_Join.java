package uk.ac.shef.inf.wop.app;

import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.client.solrj.util.ClientUtils;
import org.apache.solr.common.SolrDocument;
import uk.ac.shef.inf.wop.exporting.ProdDescTextFileExporter_Lucene;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.RecursiveTask;

public class ProdDescCorpusForBert_Join extends RecursiveTask<Integer> {

    private static final Logger LOG = Logger.getLogger(ProdDescCorpusForBert_Join.class.getName());

    private List<String> tasks;
    private int threadID;
    private String outFolder;
    private String[] maxResults;
    private SolrClient prodNameDescIndex;
    private String dataset;
    private double sample;
    private ProdDescCorpusForBert_ConcurrentSet allSelected;
    private int maxTasksPerThread;

    public ProdDescCorpusForBert_Join(int id,
                                      List<String> tasks,
                                      String outFolder,
                                      String maxResults[],
                                      SolrClient prodNameDescIndex,
                                      String dataset,
                                      double sample,
                                      ProdDescCorpusForBert_ConcurrentSet allSelected,
                                      int maxTasksPerThread) {
        this.threadID = id;
        this.tasks = tasks;
        this.outFolder = outFolder;
        this.maxResults = maxResults;
        this.prodNameDescIndex = prodNameDescIndex;
        this.sample = sample;
        this.dataset = dataset;
        this.allSelected = allSelected;
        this.maxTasksPerThread = maxTasksPerThread;
    }

    public int computeSingleWorker(List<String> tasks) {
        try {
            Set<Integer> sample_lines = new HashSet<>();
            if (sample < 1.0) {
                System.out.println("<<THREAD " + threadID + ">> Processing only sample size=" + sample + ", or " + sample_lines.size() + " records");
                List<Integer> all_lines = new ArrayList<>();

                int countAll = 0;
                for (String t : tasks) {
                    all_lines.add(countAll);
                    countAll++;
                }
                int toSelect = (int) (sample * countAll);
                Collections.shuffle(all_lines);
                sample_lines = new HashSet<>(all_lines.subList(0, toSelect));
            }

            //Gson googleJson = new Gson();
            int countRecords = 0;

            System.out.println(new Date()+"\t<<THREAD " + threadID + ">> Processing data size=" + tasks.size());
            Map<Integer, OutputStreamWriter> writers = new HashMap<>();
            for (String maxR : maxResults) {
                String outDir = outFolder + "/" + maxR;
                new File(outDir).mkdirs();

                String outFile = outDir + "/thread" + threadID + "_" + dataset + ".txt";
                OutputStreamWriter writer =
                        new OutputStreamWriter(new FileOutputStream(outFile), StandardCharsets.UTF_8);
                writers.put(Integer.valueOf(maxR), writer);
            }


            int total_selected = 0;
            for (String name : tasks) {
                if (sample_lines.size() > 0 && !sample_lines.contains(countRecords)) {
                    countRecords++;
                    continue;
                }

                //System.out.println(new Date()+"\t<<THREAD " + threadID + ">> =" + name);
                //List row_values = (ArrayList) row;

                if (name == null || name.length() == 0) {
                    System.err.println("<<THREAD " + threadID + ">> Line " + countRecords + " has no name, skip");
                    continue;
                }
                if (name.endsWith("-"))
                    name = name.substring(0, name.length() - 1).trim();
                if (name.length() == 0) {
                    countRecords++;
                    continue;
                }

                //System.out.println(new Date()+"\tRead "+ name);
                total_selected += expand(name, writers);

                //System.out.println(new Date()+"\t\tProcessed "+ name);
                //nextRecord[WOP_DESC_COL] = newDesc;
                countRecords++;

                if (countRecords % 100 == 0)
                    System.out.println("<<THREAD " + threadID + ">> " + new Date() + " \t done " + countRecords);
            }

            for (OutputStreamWriter w : writers.values())
                w.close();

            System.out.println("<<THREAD " + threadID + ">> Total records=" + countRecords);

        } catch (IOException e) {
            e.printStackTrace();
        }
        return tasks.size();

    }

    /**
     * @param productName
     */
    public int expand(String productName, Map<Integer, OutputStreamWriter> maxR_and_writers) {
        Set<Integer> maxRs = new HashSet<>(maxR_and_writers.keySet());
        SolrQuery q = createQuery(200, productName);
        QueryResponse res;
        long total = 0;
        List<String> selected = new ArrayList<>();

        try {
            res = prodNameDescIndex.query(q);
            if (res != null)
                total = res.getResults().getNumFound();
            //update results
            /*LOG.info(String.format("\t\tprocessing from %s, %d results ...",
                    productName, total));*/

            int countResults = 0;
            for (SolrDocument d : res.getResults()) {
                String docid = getStringValue(d, "id");
                if (maxRs.size() == 0)
                    break;

                String name = getStringValue(d, "name");
                if (name != null) {
                    name = removeSpaces(name);
                    if (name.toLowerCase().equalsIgnoreCase(productName.toLowerCase()))
                        continue;
                }
                String vdesc = getStringValue(d, "desc");
                vdesc = ProdDescTextFileExporter_Lucene.cleanData(vdesc);
                String[] tokens = vdesc.split("\\s+");
                if (vdesc.length() > 20 && tokens.length >= ProdDescTextFileExporter_Lucene.MIN_DESC_WORDS
                        && vdesc.length() < ProdDescTextFileExporter_Lucene.MAX_DESC_WORDS * 10) {
                    if (tokens.length > ProdDescTextFileExporter_Lucene.MAX_DESC_WORDS)
                        vdesc = StringUtils.join(tokens, " ", 0, ProdDescTextFileExporter_Lucene.MAX_DESC_WORDS);

                    countResults++;

                    if (!allSelected.contains(docid)) {
                        selected.add(vdesc);
                        allSelected.add(docid);
                    }
                }

                //check if we should dump for any 'max Results' writer
                int finish_maxR = -1;
                for (int mr : maxRs) {
                    if (countResults >= mr) {
                        finish_maxR = mr;
                        OutputStreamWriter writer = maxR_and_writers.get(mr);
                        for (String description : selected)
                            writer.write(description + "\n");
                    }
                }
                maxRs.remove(finish_maxR);
            }


        } catch (Exception e) {
            e.printStackTrace();
            System.out.println(String.format("\t\t\t error encountered, skipped due to error: %s",
                    ExceptionUtils.getFullStackTrace(e)));
        }
        return selected.size();

    }

    private static SolrQuery createQuery(int resultBatchSize, String name) {
        SolrQuery query = new SolrQuery();
        query.setQuery("desc:" + ClientUtils.escapeQueryChars(name));
        //query.setSort("random_1234", SolrQuery.ORDER.asc);
        query.setStart(0);
        query.setRows(resultBatchSize);

        return query;
    }

    private static String getStringValue(SolrDocument doc, String field) {
        Object v = doc.getFieldValue(field);
        if (v != null)
            return v.toString();
        else
            return null;
    }

    private static String removeSpaces(String v) {
        return v.replaceAll("\\s+", " ").trim();
    }

    @Override
    protected Integer compute() {
        if (this.tasks.size() > maxTasksPerThread) {
            List<ProdDescCorpusForBert_Join> subWorkers =
                    new ArrayList<>(createSubWorkers());
            for (ProdDescCorpusForBert_Join subWorker : subWorkers) {
                System.out.println("\t\tforking... thread="+subWorker.threadID);
                subWorker.fork();
            }
            return mergeResult(subWorkers);
        } else {
            System.out.println("\t\t starting worker=thread="+this.threadID);
            return computeSingleWorker(this.tasks);
        }
    }


    protected List<ProdDescCorpusForBert_Join> createSubWorkers() {
        List<ProdDescCorpusForBert_Join> subWorkers =
                new ArrayList<>();

        boolean b = false;
        List<String> splitTask1 = new ArrayList<>();
        List<String> splitTask2 = new ArrayList<>();
        for (String s : tasks) {
            if (b)
                splitTask1.add(s);
            else
                splitTask2.add(s);
            b = !b;
        }

        ProdDescCorpusForBert_Join subWorker1 = createInstance(splitTask1, this.threadID + 1);
        ProdDescCorpusForBert_Join subWorker2 = createInstance(splitTask2, this.threadID + 2);

        subWorkers.add(subWorker1);
        subWorkers.add(subWorker2);

        return subWorkers;
    }

    /**
     * NOTE: classes implementing this method must call setHashtagMap and setMaxPerThread after creating your object!!
     *
     * @param splitTasks
     * @param id
     * @return
     */
    protected ProdDescCorpusForBert_Join createInstance(List<String> splitTasks, int id) {
        ProdDescCorpusForBert_Join worker = new ProdDescCorpusForBert_Join(id,
                splitTasks,
                this.outFolder,
                this.maxResults,
                this.prodNameDescIndex,
                this.dataset,
                this.sample,
                this.allSelected,
                this.maxTasksPerThread);
        return worker;
    }
    /*{
        return new NTripleIndexerApp(id, this.solrClient, splitTasks, maxTasksPerThread, outFolder);
    }*/

    protected int mergeResult(List<ProdDescCorpusForBert_Join> workers) {
        Integer total = 0;
        for (ProdDescCorpusForBert_Join worker : workers) {
            total += worker.join();
        }
        return total;
    }
}
