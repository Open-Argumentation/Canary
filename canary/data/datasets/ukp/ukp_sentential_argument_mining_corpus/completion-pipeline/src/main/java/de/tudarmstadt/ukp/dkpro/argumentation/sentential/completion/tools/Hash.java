package de.tudarmstadt.ukp.dkpro.argumentation.sentential.completion.tools;

import java.security.MessageDigest;

public class Hash {

	public static String get(String str) {
		String hash = null;
		try {
			MessageDigest digest = MessageDigest.getInstance("MD5");
			byte[] inputBytes = str.getBytes();
			byte[] hashBytes = digest.digest(inputBytes);
			// Append bytes to string to avoid overflow
			hash = "";
			for (byte b : hashBytes) {
				hash += (String.format("%02x", b));
			}
			// Now we need to zero pad it if you actually want the full 32 chars.
			while (hash.length() < 32) {
				hash = "0" + hash;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return hash;
	}
}
