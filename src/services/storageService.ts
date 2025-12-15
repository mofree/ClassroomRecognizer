/**
 * Storage Service for persisting student data
 * Uses localStorage for client-side persistence
 */

// Define a serializable version of the Student interface
export interface SerializableStudent {
  id: string;
  name: string;
  photoUrl: string;
  descriptors: number[][]; // Convert Float32Array to number[] for serialization
  createdAt: number;
}

/**
 * Convert Student to SerializableStudent for storage
 */
export function serializeStudent(student: any): SerializableStudent {
  return {
    id: student.id,
    name: student.name,
    photoUrl: student.photoUrl,
    descriptors: student.descriptors.map((desc: Float32Array) => Array.from(desc)),
    createdAt: student.createdAt
  };
}

/**
 * Convert SerializableStudent back to Student
 */
export function deserializeStudent(serialized: SerializableStudent): any {
  return {
    id: serialized.id,
    name: serialized.name,
    photoUrl: serialized.photoUrl,
    descriptors: serialized.descriptors.map(desc => new Float32Array(desc)),
    createdAt: serialized.createdAt
  };
}

/**
 * Save students to localStorage
 */
export function saveStudentsToStorage(students: any[]): void {
  try {
    const serializableStudents = students.map(serializeStudent);
    const jsonData = JSON.stringify(serializableStudents);
    localStorage.setItem('classroomRecognizer_students', jsonData);
    console.log('Students saved to localStorage');
  } catch (error) {
    console.error('Failed to save students to localStorage:', error);
  }
}

/**
 * Load students from localStorage
 */
export function loadStudentsFromStorage(): any[] {
  try {
    const jsonData = localStorage.getItem('classroomRecognizer_students');
    if (jsonData) {
      const serializableStudents: SerializableStudent[] = JSON.parse(jsonData);
      const students = serializableStudents.map(deserializeStudent);
      console.log(`Loaded ${students.length} students from localStorage`);
      return students;
    }
    console.log('No students found in localStorage');
    return [];
  } catch (error) {
    console.error('Failed to load students from localStorage:', error);
    return [];
  }
}

/**
 * Remove all students from localStorage
 */
export function clearStudentsFromStorage(): void {
  try {
    localStorage.removeItem('classroomRecognizer_students');
    console.log('Students cleared from localStorage');
  } catch (error) {
    console.error('Failed to clear students from localStorage:', error);
  }
}

/**
 * Save recognition parameters to localStorage
 */
export function saveParamsToStorage(params: any): void {
  try {
    const jsonData = JSON.stringify(params);
    localStorage.setItem('classroomRecognizer_params', jsonData);
    console.log('Parameters saved to localStorage');
  } catch (error) {
    console.error('Failed to save parameters to localStorage:', error);
  }
}

/**
 * Load recognition parameters from localStorage
 */
export function loadParamsFromStorage(): any | null {
  try {
    const jsonData = localStorage.getItem('classroomRecognizer_params');
    if (jsonData) {
      const params = JSON.parse(jsonData);
      console.log('Parameters loaded from localStorage');
      return params;
    }
    console.log('No parameters found in localStorage');
    return null;
  } catch (error) {
    console.error('Failed to load parameters from localStorage:', error);
    return null;
  }
}