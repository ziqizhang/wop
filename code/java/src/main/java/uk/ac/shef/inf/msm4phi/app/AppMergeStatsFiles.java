package uk.ac.shef.inf.msm4phi.app;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Merge the stats files created by multiple threads and sort them
 */
public class AppMergeStatsFiles {
    public static void main(String[] args) throws IOException {
        List<String> merged = new ArrayList<>();
        String header="";
        for (File f : new File(args[0]).listFiles()){
            List<String> lines=FileUtils.readLines(f, Charset.forName("utf-8"));
            header=lines.get(0);
            lines.remove(0);
            merged.addAll(lines);
        }
        Collections.sort(merged);
        merged.add(0, header);
        PrintWriter p = new PrintWriter(args[1]);
        for(String l: merged)
            p.println(l);

        p.close();
    }
}
