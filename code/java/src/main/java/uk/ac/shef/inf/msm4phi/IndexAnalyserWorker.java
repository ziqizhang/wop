package uk.ac.shef.inf.msm4phi;

import org.apache.solr.client.solrj.SolrClient;

import java.util.*;
import java.util.concurrent.RecursiveTask;

public abstract class IndexAnalyserWorker extends RecursiveTask<Integer> {
    protected int id;
    protected SolrClient solrClient;
    protected int resultBatchSize = 10000;
    protected String outFolder;
    protected int maxTasksPerThread;

    protected Map<String, List<String>> hashtagMap;

    public IndexAnalyserWorker(int id, SolrClient solrClient,
                             String outFolder) {
        this.id=id;
        this.solrClient = solrClient;
        this.outFolder = outFolder;
    }

    public int getResultBatchSize() {
        return this.resultBatchSize;
    }

    public void setResultBatchSize(int resultBatchSize) {
        this.resultBatchSize = resultBatchSize;
    }

    public void setHashtagMap(Map<String, List<String>> hashtagMap){
        this.hashtagMap=hashtagMap;
    }
    public void setMaxTasksPerThread(int max){
        this.maxTasksPerThread=max;
    }

    @Override
    protected Integer compute() {
        if (this.hashtagMap.size() > maxTasksPerThread) {
            List<IndexAnalyserWorker> subWorkers =
                    new ArrayList<>(createSubWorkers());
            for (IndexAnalyserWorker subWorker : subWorkers)
                subWorker.fork();
            return mergeResult(subWorkers);
        } else {
            return computeSingleWorker(hashtagMap);
        }
    }


    /**
     * Query the solr backend to process tweets
     *
     * @param tasks
     * @return
     */
    protected abstract int computeSingleWorker(Map<String, List<String>> tasks);


    protected List<IndexAnalyserWorker> createSubWorkers() {
        List<IndexAnalyserWorker> subWorkers =
                new ArrayList<>();

        boolean b = false;
        Map<String, List<String>> splitTask1 = new HashMap<>();
        Map<String, List<String>> splitTask2 = new HashMap<>();
        for (Map.Entry<String, List<String>> e : hashtagMap.entrySet()) {
            if (b)
                splitTask1.put(e.getKey(), e.getValue());
            else
                splitTask2.put(e.getKey(), e.getValue());
            b = !b;
        }

        IndexAnalyserWorker subWorker1 = createInstance(splitTask1, this.id+1);
        IndexAnalyserWorker subWorker2 = createInstance(splitTask2, this.id+2);

        subWorkers.add(subWorker1);
        subWorkers.add(subWorker2);

        return subWorkers;
    }

    /**
     * NOTE: classes implementing this method must call setHashtagMap and setMaxPerThread after creating your object!!
     * @param splitTasks
     * @param id
     * @return
     */
    protected abstract IndexAnalyserWorker createInstance(Map<String, List<String>> splitTasks, int id);
    /*{
        return new IndexAnalyserWorker(id, this.solrClient, splitTasks, maxTasksPerThread, outFolder);
    }*/

    protected int mergeResult(List<IndexAnalyserWorker> workers) {
        Integer total = 0;
        for (IndexAnalyserWorker worker : workers) {
            total += worker.join();
        }
        return total;
    }

}
