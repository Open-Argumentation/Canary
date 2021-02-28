package de.tudarmstadt.ukp.dkpro.argumentation.sentential.completion.tools;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Collection;
import java.util.LinkedList;

public class TSVTools {

	public static void writeTSV(File target, Collection<SentenceEntry> entries, boolean includeSentence) {
		try {
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(target), "UTF-8"));

			bw.write("topic\tretrievedUrl\tarchivedUrl\tsentenceHash\tsentence\tannotation\tset\n");
			int i = 0;
			for (SentenceEntry e : entries) {
				bw.write(e.topic + "\t" + e.retrievedUrl + "\t" + e.archiveUrl + "\t" + e.sentenceHash + "\t"
						+ ((includeSentence) ? e.sentence : "-") + "\t" + e.annotation + "\t" + e.set + "\n");
				i++;
			}
			System.out.println("Wrote " + i + " sentence to tsv-file");
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static Collection<SentenceEntry> readTSV(File f) {
		LinkedList<SentenceEntry> results = new LinkedList<SentenceEntry>();

		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f), "UTF8"));
			String line = br.readLine();
			while ((line = br.readLine()) != null) {
				String[] tmp = line.split("\t");

				SentenceEntry entry = new SentenceEntry();
				entry.topic = tmp[0];
				entry.retrievedUrl = tmp[1];
				entry.archiveUrl = tmp[2];
				entry.sentenceHash = tmp[3];
				entry.sentence = (tmp[4].equals("-") ? null : tmp[4]);
				entry.annotation = tmp[5];
				entry.set = tmp[6];

				results.add(entry);
			}
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

		return results;
	}

}
