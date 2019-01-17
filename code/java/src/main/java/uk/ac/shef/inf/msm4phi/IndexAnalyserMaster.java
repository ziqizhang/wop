package uk.ac.shef.inf.msm4phi;

import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;

public class IndexAnalyserMaster {

    private static final Logger LOG = Logger.getLogger(IndexAnalyserMaster.class.getName());

    private Map<String, List<String>> hashtagMap;
    private int threads=1;
    private IndexAnalyserWorker worker;

    public IndexAnalyserMaster(File hashtagFile,
                               IndexAnalyserWorker worker) throws IOException {
        this.hashtagMap=Util.readHashtags(hashtagFile);
        this.worker=worker;
    }

    public void process() {
        try {

            int maxPerThread = hashtagMap.size() / threads;
            worker.setHashtagMap(this.hashtagMap);
            worker.setMaxTasksPerThread(maxPerThread);

            LOG.info(String.format("Beginning processing %d hastags on %d threads, at %s", hashtagMap.size(), threads,
                    new Date().toString()));

            ForkJoinPool forkJoinPool = new ForkJoinPool(maxPerThread);
            int total = forkJoinPool.invoke(worker);

            LOG.info(String.format("Completed %d hashtags at %s", total, new Date().toString()));

        } catch (Exception ioe) {
            StringBuilder sb = new StringBuilder("Failed to build features!");
            sb.append("\n").append(ExceptionUtils.getFullStackTrace(ioe));
            LOG.error(sb.toString());
        }

    }

    public void setThreads(int threads) {
        this.threads = threads;
    }
}
