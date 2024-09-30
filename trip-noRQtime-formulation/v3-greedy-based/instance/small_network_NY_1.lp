vertiport(jfk;lga;teb).
% distance= miles/10
distance((jfk, jfk), 0).
distance((jfk, lga), 10).
distance((jfk, teb), 19).
distance((lga, jfk), 10).
distance((lga, lga), 0).
distance((lga, teb), 9).
distance((teb, jfk), 19).
distance((teb, lga), 9).
distance((teb, teb), 0).
flight_time((jfk, jfk), 0).
flight_time((jfk, lga), 6).
flight_time((jfk, teb), 11).
flight_time((lga, jfk), 6).
flight_time((lga, lga), 0).
flight_time((lga, teb), 5).
flight_time((teb, jfk), 11).
flight_time((teb, lga), 5).
flight_time((teb, teb), 0).
charge_time((jfk, jfk), 0).
charge_time((jfk, lga), 6).
charge_time((jfk, teb), 11).
charge_time((lga, jfk), 6).
charge_time((lga, lga), 0).
charge_time((lga, teb), 5).
charge_time((teb, jfk), 11).
charge_time((teb, lga), 5).
charge_time((teb, teb), 0).
edge(jfk,jfk).
edge(jfk,lga).
edge(jfk,teb).
edge(lga,jfk).
edge(lga,lga).
edge(lga,teb).
edge(teb,jfk).
edge(teb,lga).
edge(teb,teb).
