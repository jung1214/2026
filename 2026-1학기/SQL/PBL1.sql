drop database if exists studyroomdb;
create database if not exists studyroomdb;
use studyroomdb;

CREATE TABLE 학생 ( 
  학번 INT NOT NULL UNIQUE, 
  학과 VARCHAR(50) NOT NULL, 
  이름 VARCHAR(20) NOT NULL, 
  전화번호 VARCHAR(20) NOT NULL UNIQUE, 
  이메일 VARCHAR(100) UNIQUE, 
  PRIMARY KEY (학번) 
);

CREATE TABLE 스터디룸 ( 
  스터디룸ID INT NOT NULL UNIQUE, 
  스터디룸명 VARCHAR(50) NOT NULL, 
  위치 VARCHAR(50) NOT NULL, 
  수용인원 CHAR(10), 
  이용요금 CHAR(10), 
  보드종류 VARCHAR(20), 
  PRIMARY KEY (스터디룸ID) 
);
CREATE TABLE 예약 ( 
  예약번호 INT NOT NULL UNIQUE, 
  스터디룸ID INT NOT NULL, 
  학번 INT NOT NULL, 
  예약날짜 DATE NOT NULL, 
  시작시간 TIME NOT NULL, 
  종료시간 TIME NOT NULL, 
  예약신청일시 DATETIME NOT NULL, 
  예약상태 BOOLEAN NOT NULL DEFAULT TRUE, 
  PRIMARY KEY (예약번호), 
  FOREIGN KEY (학번) REFERENCES 학생(학번), 
  FOREIGN KEY (스터디룸ID) REFERENCES 스터디룸(스터디룸ID) );
  
  CREATE TABLE 예약취소 ( 
  취소번호 INT NOT NULL UNIQUE, 
  예약번호 INT NOT NULL UNIQUE, 
  취소일시 DATETIME NOT NULL, 
  취소사유 VARCHAR(200), 
  환불여부 VARCHAR(10), 
  PRIMARY KEY (취소번호), 
  FOREIGN KEY (예약번호) REFERENCES 예약(예약번호) 
);
CREATE TABLE 결제 ( 
  결제번호 INT NOT NULL, 
  예약번호 INT NOT NULL, 
  결제여부 VARCHAR(10), 
  결제일시 BIGINT, 
  PRIMARY KEY (결제번호), 
  FOREIGN KEY (예약번호) REFERENCES 예약(예약번호) 
); 
 
CREATE TABLE 룸이벤트 ( 
  이벤트번호 INT AUTO_INCREMENT NOT NULL, 
  이벤트시작 BIGINT NOT NULL, 
  이벤트끝 BIGINT NOT NULL, 
  이벤트내용 VARCHAR(200), 
  스터디룸ID INT NOT NULL, 
  PRIMARY KEY (이벤트번호), 
  FOREIGN KEY (스터디룸ID) REFERENCES 스터디룸(스터디룸ID) 
);

INSERT INTO 학생 VALUES 
(92514710, '인공지능', '한예은', '010-1234-5678', 'minsu@uni.ac.kr'), 
(92488890, '전자공학과', '이서연', '010-2345-6789', 'seoyeon@uni.ac.kr'), 
(92217231, '경영학과', '박지훈', '010-3456-7890', 'jihun@uni.ac.kr'), 
(92514643,'인공지능','정서현','010-8009-2101','JUNGSEO@uni.ac.kr'),  
(92308666,'인공지능','이수현','010-8671-4505','suhyeon040526@gmail.com'); 
 
INSERT INTO 스터디룸 VALUES 
(1, 'Co-working Room1', '6층', '5명', '무료', '없음'), 
(2, 'Co-working Room2', '6층', '5명', '무료', '없음'), 
(3, 'Co-working Room3', '6층', '5명', '무료', '없음'), 
(4, 'Co-working Room4', '6층', '5명', '무료', '없음'), 
(5, 'Co-working Room5', '6층', '5명', '무료', '없음'), 
(6, 'Co-working Room6', '6층', '6명', '1000원', '없음'), 
(7, 'Co-working Room7', '6층', '9명', '1000원', '없음'), 
(8, 'Nostoi 01', '도서관 2층', '11명', '무료', '전자칠판'), 
(9, 'Mousai 02', '도서관 2층', '11명', '무료', '전자칠판'), 
(10, 'Siren 03', '도서관 2층', '8명', '1000원','전자칠판&화이트보드'); 
 
INSERT INTO 예약 VALUES 
(260410001, 6, 92514710, '2026-04-10', '10:00:00', '12:30:00', '2026-04-07 16:23:14', TRUE), 
(260411001, 2, 92488890, '2026-04-11', '13:00:00', '15:00:00', '2026-04-07 16:10:00', TRUE ), 
(260412001, 10, 92217231, '2026-04-12', '09:00:00', '11:00:00', '2026-04-07 16:20:56', 
FALSE), 
(260422001, 8, 92514643, '2026-04-22', '16:30:00', '20:00:00', '2026-04-22 12:42:12', TRUE), 
(260526001, 5, 92308666,'2026-05-26','10:00:00','11:00:00','2026-04-10 14:00:00', TRUE), 
(260526002, 5, 92308666, '2026-05-26','11:00:00','12:00:00','2026-04-10 14:44:02', TRUE), 
(260526003, 5, 92308666, '2026-05-26','12:00:00','13:00:00','2026-04-10 14:00:33', FALSE); 

INSERT INTO 결제 VALUES 
(5001, 260410001, '완료', '1775545260'), 
(5003, 260412001, '환불', '1775546460'); 
 
INSERT INTO 예약취소 VALUES 
(9001, 260412001, '2026-04-07 16:25:41', '개인 사정', '환불완료'), 
(9002, 260526003, '2026-04-10 14:05:06', '개인 사정', '확인중'); 
 
INSERT INTO 룸이벤트(이벤트시작, 이벤트끝, 이벤트내용, 스터디룸ID) VALUES 
(1778338800, 1778425199, '어린이날 휴무', 1 ), 
(1778338800, 1778425199, '어린이날 휴무', 2 ), 
(1778338800, 1778425199, '어린이날 휴무', 3 ), 
(1778338800, 1778425199, '어린이날 휴무', 4 ), 
(1778338800, 1778425199, '어린이날 휴무', 5 ), 
(1778338800, 1778425199, '어린이날 휴무', 6 ), 
(1778338800, 1778425199, '어린이날 휴무', 7 ), 
(1778338800, 1778425199, '어린이날 휴무', 8 ), 
(1778338800, 1778425199, '어린이날 휴무', 9 ), 
(1778338800, 1778425199, '어린이날 휴무', 10 ), 
(1777042800, 1777215599, '공사', 3); 
 
SELECT * FROM 학생; 
SELECT * FROM 스터디룸; 
SELECT * FROM 예약; 
SELECT * FROM 결제; 
SELECT * FROM 예약취소; 
SELECT * FROM 룸이벤트; 
