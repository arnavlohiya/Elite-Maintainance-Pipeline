"use client";

import Link from "next/link";
import { signIn, signOut, useSession } from "next-auth/react";

export default function Navbar() {
  const { data: session } = useSession();

  return (
    <nav className="p-4 flex justify-between bg-gray-900 text-white">
      <div className="flex gap-4">
        <Link href="/">Home</Link>
        <Link href="/upload">Upload</Link>
        <Link href="/jobs">Jobs</Link>
      </div>

      <div>
        {!session && (
          <button onClick={() => signIn("google")} className="px-3 py-1 bg-blue-600 rounded">
            Login with Google
          </button>
        )}
        {session && (
          <button onClick={() => signOut()} className="px-3 py-1 bg-red-600 rounded">
            Logout
          </button>
        )}
      </div>
    </nav>
  );
}
